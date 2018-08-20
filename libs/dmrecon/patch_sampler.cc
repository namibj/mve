/*
 * Copyright (C) 2015, Ronny Klowsky, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "math/defines.h"
#include "math/matrix.h"
#include "math/vector.h"
#include "dmrecon/defines.h"
#include "dmrecon/mvs_tools.h"
#include "dmrecon/patch_sampler.h"

MVS_NAMESPACE_BEGIN

PatchSampler::PatchSampler(std::vector<SingleView::Ptr> const& _views,
	const int refViewIdx, const int filterWidth,
    int _x, int _y, float _depth, float _dzI, float _dzJ)
    : views(_views)
	, refVIdx(refViewIdx)
    , midPix(_x,_y)
    , masterMeanCol(0.f)
    , depth(_depth)
    , dzI(_dzI)
    , dzJ(_dzJ)
    , success(views.size(), false)
{
	SingleView::Ptr refV(views[refVIdx]);
    mve::ByteImage::ConstPtr masterImg(refV->getScaledImg());

	offset = filterWidth / 2;
	nrSamples = sqr(filterWidth);

    /* initialize arrays */
    patchPoints.resize(nrSamples);
    masterColorSamples.resize(nrSamples);
    masterViewDirs.resize(nrSamples);

    /* compute patch position and check if it's valid */
    math::Vec2i h;
    h[0] = h[1] = offset;
    topLeft = midPix - h;
    bottomRight = midPix + h;
    if (topLeft[0] < 0 || topLeft[1] < 0
        || bottomRight[0] > masterImg->width()-1
        || bottomRight[1] > masterImg->height()-1)
        return;

    /* initialize viewing rays from master view */
    std::size_t count = 0;
    for (int j = topLeft[1]; j <= bottomRight[1]; ++j)
        for (int i = topLeft[0]; i <= bottomRight[0]; ++i)
            masterViewDirs[count++] = refV->viewRayScaled(i, j);

    /* initialize master color samples and 3d patch points */
	success[refVIdx] = true;
    computeMasterSamples();
    computePatchPoints();
}

void
PatchSampler::fastColAndDeriv(std::size_t v, Samples& color, Samples& deriv)
{
    success[v] = false;
	SingleView::Ptr refV = views[refVIdx];

    PixelCoords& imgPos = neighPosSamples[v];
    imgPos.resize(nrSamples);

    math::Vec3f const& p0 = patchPoints[nrSamples/2];
    /* compute pixel prints and decide on which MipMap-Level to draw
       the samples */
    float mfp = refV->footPrintScaled(p0);
    float nfp = views[v]->footPrint(p0);
    if (mfp <= 0.f) {
        std::cerr << "Error in getFastColAndDerivSamples! "
                  << "footprint in master view: " << mfp << std::endl;
        throw std::out_of_range("Negative pixel footprint");
    }
    if (nfp <= 0.f)
        return;
    float ratio = nfp / mfp;
    int mmLevel = 0;
    while (ratio < 0.5f) {
        ++mmLevel;
        ratio *= 2.f;
    }
    mmLevel = views[v]->clampLevel(mmLevel);

    /* compute step size for derivative */
    math::Vec3f p1(p0 + masterViewDirs[nrSamples/2]);
    float d = (views[v]->worldToScreen(p1, mmLevel)
        - views[v]->worldToScreen(patchPoints[12], mmLevel)).norm();
    if (!(d > 0.f)) {
        return;
    }
    stepSize[v] = 1.f / d;

    /* request according undistorted color image */
    mve::ByteImage::ConstPtr img(views[v]->getPyramidImg(mmLevel));
    int const w = img->width();
    int const h = img->height();

    /* compute image position and gradient direction for each sample
       point in neighbor image v */
    std::vector<math::Vec2f> gradDir(nrSamples);
    for (std::size_t i = 0; i < nrSamples; ++i)
    {
        math::Vec3f p0(patchPoints[i]);
        math::Vec3f p1(patchPoints[i] + masterViewDirs[i] * stepSize[v]);
        imgPos[i] = views[v]->worldToScreen(p0, mmLevel);
        // imgPos should be away from image border
        if (!(imgPos[i][0] > 0 && imgPos[i][0] < w-1 &&
                imgPos[i][1] > 0 && imgPos[i][1] < h-1)) {
            return;
        }
        gradDir[i] = views[v]->worldToScreen(p1, mmLevel) - imgPos[i];
    }

    /* draw the samples in the image */
    color.resize(nrSamples, math::Vec3f(0.f));
    deriv.resize(nrSamples, math::Vec3f(0.f));
    colAndExactDeriv(*img, imgPos, gradDir, color, deriv);

    /* normalize the gradient */
    for (std::size_t i = 0; i < nrSamples; ++i)
        deriv[i] /= stepSize[v];

    success[v] = true;
}

float
PatchSampler::getFastNCC(std::size_t v)
{
    if (neighColorSamples[v].empty())
        computeNeighColorSamples(v);
    if (!success[v])
        return -1.f;
	assert(success[refVIdx]);
    math::Vec3f meanY(0.f);
    for (std::size_t i = 0; i < nrSamples; ++i)
        meanY += neighColorSamples[v][i];
    meanY /= (float) nrSamples;

    float sqrDevY = 0.f;
    float devXY = 0.f;
    for (std::size_t i = 0; i < nrSamples; ++i)
    {
        sqrDevY += (neighColorSamples[v][i] - meanY).square_norm();
        // Note: master color samples are normalized!
        devXY += (masterColorSamples[i] - meanX)
            .dot(neighColorSamples[v][i] - meanY);
    }
    float tmp = sqrt(sqrDevX * sqrDevY);
    assert(!MATH_ISNAN(tmp) && !MATH_ISNAN(devXY));
    if (tmp > 0)
        return (devXY / tmp);
    else
        return -1.f;
}

float
PatchSampler::getSSD(std::size_t v, math::Vec3f const& cs)
{
    if (neighColorSamples[v].empty())
        computeNeighColorSamples(v);
    if (!success[v])
        return -1.f;

    float sum = 0.f;
    for (std::size_t i = 0; i < nrSamples; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            float diff = cs[c] * neighColorSamples[v][i][c] -
                masterColorSamples[i][c];
            sum += diff * diff;
        }
    }
    return sum;
}

math::Vec3f
PatchSampler::getPatchNormal() const
{
    std::size_t right = nrSamples/2 + offset;
    std::size_t left = nrSamples/2 - offset;
    std::size_t top = offset;
    std::size_t bottom = nrSamples - 1 - offset;

    math::Vec3f a(patchPoints[right] - patchPoints[left]);
    math::Vec3f b(patchPoints[top] - patchPoints[bottom]);
    math::Vec3f normal(a.cross(b));
    normal.normalize();

    return normal;
}

void
PatchSampler::update(float newDepth, float newDzI, float newDzJ)
{
    success.clear();
    success.resize(views.size(), false);
    depth = newDepth;
    dzI = newDzI;
    dzJ = newDzJ;
	success[refVIdx] = true;
    computePatchPoints();
    neighColorSamples.clear();
    neighDerivSamples.clear();
    neighPosSamples.clear();
}

void
PatchSampler::computePatchPoints()
{
	SingleView::Ptr refV = views[refVIdx];

    unsigned int count = 0;
    for (int j = topLeft[1]; j <= bottomRight[1]; ++j)
    {
        for (int i = topLeft[0]; i <= bottomRight[0]; ++i)
        {
            float tmpDepth = depth + (i - midPix[0]) * dzI +
                (j - midPix[1]) * dzJ;
            if (tmpDepth <= 0.f)
            {
				success[refVIdx] = false;
                return;
            }
            patchPoints[count] = refV->camPos + tmpDepth *
                masterViewDirs[count];
            ++count;
        }
    }
}

float
PatchSampler::getSAD(std::size_t v, math::Vec3f const& cs)
{
	if (neighColorSamples[v].empty())
		computeNeighColorSamples(v);
	if (!success[v])
		return -1.f;

	float sum = 0.f;
	for (std::size_t i = 0; i < nrSamples; ++i) {
		for (int c = 0; c < 3; ++c) {
			sum += std::abs(cs[c] * neighColorSamples[v][i][c] -
				masterColorSamples[i][c]);
		}
	}
	return sum;
}

bool PatchSampler::getCenterVariance(float &variance, const IndexSet &viewSet)
{
	std::vector<math::Vec3f> normalizedColors;
	normalizedColors.reserve(viewSet.size() + 1);

	// get normalized colors of center pixels of images from viewSet
	math::Vec3f normalizedColor;
	for (IndexSet::const_iterator id = viewSet.begin(); id != viewSet.end(); ++id)
	{
		const int viewIdx = *id;
		if (getNormalizedCenterColor(normalizedColor, viewIdx))
			normalizedColors.push_back(normalizedColor);
	}
	if (getNormalizedCenterColor(normalizedColor, refVIdx))
		normalizedColors.push_back(normalizedColor);

	// get & check number of normalized colors
	const std::size_t colorCount = normalizedColors.size();
	if (colorCount <= 3) // magic number 42
		return false;


	// compute mean & variance of normalized colors
	math::Vec3f meanOfCenter(0.0f, 0.0f, 0.0f);
	for (std::size_t i = 0; i < colorCount; ++i)
		meanOfCenter += normalizedColors[i];
	meanOfCenter /= static_cast<float>(colorCount);

	variance = 0.0f;
	for (std::size_t i = 0; i < colorCount; ++i)
		variance += (normalizedColors[i] - meanOfCenter).square_norm();
	variance /= static_cast<float>(colorCount);

	return true;
}

bool PatchSampler::getNormalizedCenterColor(math::Vec3f &normalizedColor,
	const std::size_t viewIdx)
{
	// get proper image
	int mipmapLevel;
	mve::ByteImage::ConstPtr img = getNeighborImage(mipmapLevel, viewIdx);
	if (!img)
		return false;

	// get mean color & variance
	math::Vec3f mean;
	float variance;
	if (!getMean(mean, viewIdx))
		return false;
	if (!getVariance(variance, mean, viewIdx))
		return false;

	// get center color
	const math::Vec3f &centerWS = patchPoints[nrSamples/2];
	const math::Vec2f centerVS = views[viewIdx]->worldToScreen(centerWS, mipmapLevel);
	const math::Vec3f color = getXYZColorAtPos(*img, centerVS);

	// normalize color
	normalizedColor = (color - mean) / (variance); // todo magic constant 42
	return true;
}

float PatchSampler::getNCC(std::size_t u, std::size_t v)
{
	math::Vec3f meanX;
	math::Vec3f meanY;
	float varX;
	float varY;

	if (!getMean(meanX, u))
		return -1.f;
	getVariance(varX, meanX, u);

	if (!getMean(meanY, v))
		return -1.f;
	getVariance(varY, meanY, v);

	// cross variance
	float crossVariance = 0.f;
	for (std::size_t i = 0; i < nrSamples; ++i)
		crossVariance += (neighColorSamples[u][i] - meanX).dot(neighColorSamples[v][i] - meanY);
	crossVariance /= nrSamples;

	// cross correlation
	const float denominator = sqrt(varX * varY);
	if (denominator > 0)
		return (crossVariance / denominator);
	else
		return -1.f;
}

bool PatchSampler::getMean(math::Vec3f &mean,
	const std::size_t v)
{
	if (neighColorSamples[v].empty())
		computeNeighColorSamples(v);
	if (!success[v])
		return false;

	mean = math::Vec3f(0.f, 0.f, 0.f);
	for (std::size_t i = 0; i < nrSamples; ++i)
		mean += neighColorSamples[v][i];
	mean /= nrSamples;

	return true;
}

bool PatchSampler::getVariance(float &variance,
	const math::Vec3f &mean, const std::size_t v)
{
	if (neighColorSamples[v].empty())
		computeNeighColorSamples(v);
	if (!success[v])
		return false;

	variance = 0.f;
	for (std::size_t i = 0; i < nrSamples; ++i)
		variance += (neighColorSamples[v][i] - mean).square_norm();
	variance /= nrSamples;

	return true;
}

mve::ByteImage::ConstPtr PatchSampler::getNeighborImage(int &mipmapLevel,
	const std::size_t v) const
{
	/* compute pixel prints and decide on which MipMap-Level to draw
	   the samples */
	const math::Vec3f &p0 = patchPoints[nrSamples/2];
	SingleView::Ptr refV = views[refVIdx];

	const float mfp = refV->footPrintScaled(p0);
	const float nfp = views[v]->footPrint(p0);
	if (mfp <= 0.f)
	{
		std::cerr << "Error in computeNeighColorSamples! "
				  << "footprint in master view: " << mfp << std::endl;
		throw std::out_of_range("Negative pixel print");
	}
	if (nfp <= 0.f)
		return nullptr;

	float ratio = nfp / mfp;

	mipmapLevel = 0;
	while (ratio < 0.5f)
	{
		++mipmapLevel;
		ratio *= 2.f;
	}
	mipmapLevel = views[v]->clampLevel(mipmapLevel);
	mve::ByteImage::ConstPtr img(views[v]->getPyramidImg(mipmapLevel));

	return img;
}

void
PatchSampler::computeNeighColorSamples(std::size_t v)
{
	Samples &colors = neighColorSamples[v];
	PixelCoords &imgPositions = neighPosSamples[v];
	success[v] = false;

	int mipmapLevel;
	mve::ByteImage::ConstPtr img = getNeighborImage(mipmapLevel, v);
	if (!img)
		return;

	colors.resize(nrSamples);
	imgPositions.resize(nrSamples);
	int const w = img->width();
	int const h = img->height();

	for (std::size_t i = 0; i < nrSamples; ++i)
	{
		imgPositions[i] = views[v]->worldToScreen(patchPoints[i], mipmapLevel);
		// imgPos should be away from image border
		if (!(imgPositions[i][0] > 0 && imgPositions[i][0] < w-1 &&
			imgPositions[i][1] > 0 && imgPositions[i][1] < h-1))
				return;
	}
	getXYZColorAtPos(*img, imgPositions, &colors);
	success[v] = true;
}

void PatchSampler::computeMasterSamples()
{
	SingleView::Ptr refV = views[refVIdx];
	mve::ByteImage::ConstPtr img(refV->getScaledImg());

	/* draw color samples from image and compute mean color */
	std::size_t count = 0;
	std::vector<math::Vec2i> imgPos(nrSamples);
	for (int j = topLeft[1]; j <= bottomRight[1]; ++j)
		for (int i = topLeft[0]; i <= bottomRight[0]; ++i)
		{
			imgPos[count][0] = i;
			imgPos[count][1] = j;
			++count;
		}
	getXYZColorAtPix(*img, imgPos, &masterColorSamples);

	masterMeanCol = 0.f;
	for (std::size_t i = 0; i < nrSamples; ++i)
		for (int c = 0; c < 3; ++c)
		{
			assert(masterColorSamples[i][c] >= 0 &&
				masterColorSamples[i][c] <= 1);
			masterMeanCol += masterColorSamples[i][c];
		}

	masterMeanCol /= 3.f * nrSamples;
	if (masterMeanCol < 0.01f || masterMeanCol > 0.99f) {
		success[refVIdx] = false;
		return;
	}

	meanX.fill(0.f);

	/* normalize master samples so that average intensity over all
	   channels is 1 and compute mean color afterwards */
	for (std::size_t i = 0; i < nrSamples; ++i)
	{
		masterColorSamples[i] /= masterMeanCol;
		meanX += masterColorSamples[i];
	}
	meanX /= nrSamples;
	sqrDevX = 0.f;

	/* compute variance (independent from actual mean) */
	for (std::size_t i = 0; i < nrSamples; ++i)
		sqrDevX += (masterColorSamples[i] - meanX).square_norm();
}

PatchSampler::Ptr PatchSampler::create(const std::vector<SingleView::Ptr> &views,
	const int refViewIdx, const int filterWidth,
	int x, int y, float depth, float dzI, float dzJ)
{
	// try to create the sampler
	PatchSampler::Ptr ptr(new PatchSampler(views, refViewIdx, filterWidth, x, y, depth, dzI, dzJ));

	// could the sampler be initialized properly?
	if (!ptr->success[refViewIdx])
		return nullptr;
	else
		return ptr;
}


MVS_NAMESPACE_END
