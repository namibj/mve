/*
 * Copyright (C) 2015, Ronny Klowsky, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <ctime>

#include "math/vector.h"
#include "math/functions.h"
#include "math/octree_tools.h"
#include "mve/image.h"
#include "mve/image_tools.h"
#include "util/file_system.h"
#include "util/strings.h"
#include "dmrecon/settings.h"
#include "dmrecon/dmrecon.h"
#include "dmrecon/global_view_selection.h"

MVS_NAMESPACE_BEGIN

DMRecon::DMRecon(mve::Scene::Ptr _scene, Settings const& _settings)
    : scene(_scene)
    , settings(_settings)
{
    mve::Scene::ViewList const& mve_views(scene->get_views());

    /* Check if master image exists */
    if (settings.refViewNr >= mve_views.size())
        throw std::invalid_argument("Master view index out of bounds");

    /* Check for meaningful scale factor */
    if (settings.scale < 0.f)
        throw std::invalid_argument("Invalid scale factor");

    /* Check if image embedding is set. */
    if (settings.imageEmbedding.empty())
        throw std::invalid_argument("Invalid image embedding");

    // TODO: Implement more sanity checks on the settings!

    /* Fetch bundle file. */
    try
    {
        this->bundle = this->scene->get_bundle();
    }
    catch (std::exception& e)
    {
        throw std::runtime_error(std::string("Error reading bundle file: ")
              + e.what());
    }

    /* Create list of SingleView pointers from MVE views. */
    views.resize(mve_views.size());
    for (std::size_t i = 0; i < mve_views.size(); ++i)
    {
        if (mve_views[i] == nullptr || !mve_views[i]->is_camera_valid() ||
            !mve_views[i]->has_image(this->settings.imageEmbedding,
            mve::IMAGE_TYPE_UINT8))
            continue;
        views[i] = mvs::SingleView::create(scene, mve_views[i],
            this->settings.imageEmbedding);
    }

    SingleView::Ptr refV = views[settings.refViewNr];
    if (refV == nullptr)
        throw std::invalid_argument("Invalid master view");

    /* Prepare reconstruction */
	refV->loadColorImage(settings.scale);
	refV->prepareMasterView(settings.scale, settings.keepViewIndicesPerPixel, settings.nrReconNeighbors + 1);
    mve::ByteImage::ConstPtr scaled_img = refV->getScaledImg();
    this->width = scaled_img->width();
    this->height = scaled_img->height();

    if (!settings.quiet)
        std::cout << "scaled image size: " << this->width << " x "
                  << this->height << std::endl;
}

void
DMRecon::start()
{
    try
    {
        progress.start_time = std::time(nullptr);

        analyzeFeatures();
        globalViewSelection();
        processFeatures();
        processQueue();

        if (progress.cancelled)
        {
            progress.status = RECON_CANCELLED;
            return;
        }

        progress.status = RECON_SAVING;
        SingleView::Ptr refV(views[settings.refViewNr]);
        if (settings.writePlyFile)
        {
            if (!settings.quiet)
                std::cout << "Saving ply file as "
                    << settings.plyPath << "/"
                    << refV->createFileName(settings.scale)
                    << ".ply" << std::endl;
            refV->saveReconAsPly(settings.plyPath, settings.scale);
        }

        // Save images to view
        mve::View::Ptr view = refV->getMVEView();
		std::string scale = util::string::get(settings.scale);

		std::string name("depth-L");
		name += scale;
        view->set_image(refV->depthImg, name);

        if (settings.keepDzMap)
        {
            name = "dz-L";
			name += scale;
            view->set_image(refV->dzImg, name);
        }

        if (settings.keepConfidenceMap)
        {
            name = "conf-L";
			name += scale;
            view->set_image(refV->confImg, name);
        }

        if (settings.scale != 0)
        {
            name = "undist-L";
			name += scale;
            view->set_image(refV->getScaledImg()->duplicate(), name);
        }

		if (settings.keepViewIndicesPerPixel)
		{
			name = "views-L";
			name += scale;
			view->set_image(refV->viewIndicesImg, name);
		}

        progress.status = RECON_IDLE;

        /* Output percentage of filled pixels */
        {
            int nrPix = this->width * this->height;
            float percent = (float) progress.filled / (float) nrPix;
            if (!settings.quiet)
                std::cout << "Filled " << progress.filled << " pixels, i.e. "
                          << util::string::get_fixed(percent * 100.f, 1)
                          << " %." << std::endl;
        }

        /* Output required time to process the image */
        size_t mvs_time = std::time(nullptr) - progress.start_time;
        if (!settings.quiet)
            std::cout << "MVS took " << mvs_time << " seconds." << std::endl;
    }
    catch (util::Exception e)
    {
        if (!settings.quiet)
            std::cout << "Reconstruction failed: " << e << std::endl;

        progress.status = RECON_CANCELLED;
        return;
    }
}

/*
 * Attach features that are visible in the reference view (according to
 * the bundle) to all other views if inside the frustum.
 */
void
DMRecon::analyzeFeatures()
{
    progress.status = RECON_FEATURES;

    SingleView::ConstPtr refV = views[settings.refViewNr];
    mve::Bundle::Features const& features = bundle->get_features();
    for (std::size_t i = 0; i < features.size() && !progress.cancelled; ++i)
    {
        if (!features[i].contains_view_id(settings.refViewNr))
            continue;

        math::Vec3f featurePos(features[i].pos);
        if (!refV->pointInFrustum(featurePos))
            continue;

        if (!math::geom::point_box_overlap(featurePos,
            this->settings.aabbMin, this->settings.aabbMax))
            continue;

        for (std::size_t j = 0; j < features[i].refs.size(); ++j)
        {
            int view_id = features[i].refs[j].view_id;
            if (view_id < 0 || view_id >= static_cast<int>(views.size())
                || views[view_id] == nullptr)
                continue;
            if (views[view_id]->pointInFrustum(featurePos))
                views[view_id]->addFeature(i);
        }
    }
}

void
DMRecon::globalViewSelection()
{
    progress.status = RECON_GLOBALVS;
    if (progress.cancelled)
        return;

    /* Perform global view selection. */
    GlobalViewSelection globalVS(views, bundle->get_features(), settings);
    globalVS.performVS();
    neighViews = globalVS.getSelectedIDs();

    if (neighViews.empty())
        throw std::runtime_error("Global View Selection failed");

    /* Print result of global view selection. */
    if (!settings.quiet)
    {
        std::cout << "Global View Selection:";
        for (IndexSet::const_iterator iter = neighViews.begin();
            iter != neighViews.end(); ++iter)
            std::cout << " " << *iter;
        std::cout << std::endl;
    }

    /* Load selected images. */
    if (!settings.quiet)
        std::cout << "Loading color images..." << std::endl;
    for (IndexSet::const_iterator iter = neighViews.begin();
        iter != neighViews.end() && !progress.cancelled; ++iter)
        views[*iter]->loadColorImage(0);
}

void
DMRecon::processFeatures()
{
    progress.status = RECON_FEATURES;
    if (progress.cancelled)
        return;
    SingleView::Ptr refV = views[settings.refViewNr];
    mve::Bundle::Features const& features = bundle->get_features();

    if (!settings.quiet)
        std::cout << "Processing " << features.size()
            << " features..." << std::endl;

    std::size_t success = 0;
    std::size_t processed = 0;
    for (std::size_t i = 0; i < features.size() && !progress.cancelled; ++i)
    {
        /*
         * Use feature if visible in reference view or
         * at least one neighboring view.
         */
        bool useFeature = false;
        if (features[i].contains_view_id(settings.refViewNr))
            useFeature = true;

        for (IndexSet::const_iterator id = neighViews.begin();
            useFeature == false && id != neighViews.end(); ++id)
        {
            if (features[i].contains_view_id(*id))
                useFeature = true;
        }
        if (!useFeature)
            continue;

        math::Vec3f featPos(features[i].pos);
        if (!refV->pointInFrustum(featPos))
            continue;

        /* Check if feature is inside AABB. */
        if (!math::geom::point_box_overlap(featPos,
            this->settings.aabbMin, this->settings.aabbMax))
            continue;

        /* Start processing the feature. */
		++processed;

		const math::Vec2f pixPosF = refV->worldToScreenScaled(featPos);
		const int x = math::round(pixPosF[0]);
		const int y = math::round(pixPosF[1]);

		const float initDepth = (featPos - refV->camPos).norm();
        PatchOptimization patch(views, settings, x, y, initDepth,
            0.f, 0.f, neighViews, IndexSet());
        patch.doAutoOptimization();

		// valid result?
		const float newConfidence = patch.computeConfidence();
		if (newConfidence <= 0.0f)
            continue;
		++success;

		// worse than previous result?
		const int centerIdx = y * this->width + x;
		const float refVConfidence = refV->confImg->at(centerIdx);
		if (newConfidence <= refVConfidence)
			continue;

		if (refVConfidence <= 0)
			++progress.filled;

		updateReferenceView(refV, patch, newConfidence, centerIdx, settings.keepViewIndicesPerPixel);

		const QueueData data(x, y, newConfidence, patch.getDepth(), patch.getDzI(), patch.getDzJ(), patch.getLocalViewIDs());
		prQueue.push(data);
    }
    if (!settings.quiet)
        std::cout << "Processed " << processed << " features, from which "
                  << success << " succeeded optimization." << std::endl;
}

void
DMRecon::processQueue()
{
    progress.status = RECON_QUEUE;
    if (progress.cancelled)  return;

    SingleView::Ptr refV = this->views[settings.refViewNr];

    if (!settings.quiet)
        std::cout << "Process queue ..." << std::endl;

    size_t count = 0, lastStatus = 1;
    progress.queueSize = prQueue.size();
    if (!settings.quiet)
        std::cout << "Count: " << std::setw(8) << count
                  << "  filled: " << std::setw(8) << progress.filled
                  << "  Queue: " << std::setw(8) << progress.queueSize
                  << std::endl;
    lastStatus = progress.filled;

    while (!prQueue.empty() && !progress.cancelled)
    {
        progress.queueSize = prQueue.size();
        if ((progress.filled % 1000 == 0) && (progress.filled != lastStatus))
        {
            if (!settings.quiet)
                std::cout << "Count: " << std::setw(8) << count
                          << "  filled: " << std::setw(8) << progress.filled
                          << "  Queue: " << std::setw(8) << progress.queueSize
                          << std::endl;
            lastStatus = progress.filled;
        }

		QueueData startEstimate = prQueue.top();
        prQueue.pop();
		++count;
		const int centerIdx = startEstimate.getLinearIdx(this->width);

		// next queue candidate is already worse than previous result at pixel index?
		mve::Image<float>::Ptr confImg = refV->confImg;
		const float refVConfidence = confImg->at(centerIdx);
		const float startConfidence = startEstimate.confidence;
		if (refVConfidence > startConfidence)
			continue;

		// optimize next best hypothesis
		const float x = startEstimate.x;
		const float y = startEstimate.y;
		PatchOptimization patch(views, settings, x, y, startEstimate.depth,
			startEstimate.dz_i, startEstimate.dz_j, neighViews, startEstimate.localViewIDs);
		patch.doAutoOptimization();

		// invalid result?
		const float newConfidence = patch.computeConfidence();
		if (newConfidence <= 0.0f)
			continue;
		// worse than previous result?
		if (newConfidence <= refVConfidence)
			continue;

		// filled pixel for the first time?
		if (refVConfidence <= 0)
            ++progress.filled;

		updateReferenceView(refV, patch, newConfidence, centerIdx, settings.keepViewIndicesPerPixel);

		// extend queue to neighbors
		{
			const float minConfidence = newConfidence - 0.05f;
			QueueData propagation(startEstimate.y, startEstimate.x,
				newConfidence, patch.getDepth(), patch.getDzI(), patch.getDzJ(), patch.getLocalViewIDs());

			// left
			propagation.x = startEstimate.x - 1;
			propagation.y = startEstimate.y;
			const int leftIndex = propagation.getLinearIdx(this->width);
			if (refV->confImg->at(leftIndex) < minConfidence  || refV->confImg->at(leftIndex) == 0.f)
				prQueue.push(propagation);

			// right
			propagation.x = startEstimate.x + 1;
			propagation.y = startEstimate.y;
			const int rightIndex = propagation.getLinearIdx(this->width);
			if (refV->confImg->at(rightIndex) < minConfidence || refV->confImg->at(rightIndex) == 0.f)
				prQueue.push(propagation);

			// top
			propagation.x = startEstimate.x;
			propagation.y = startEstimate.y - 1;
			const int topIndex = propagation.getLinearIdx(this->width);
			if (refV->confImg->at(topIndex) < minConfidence || refV->confImg->at(topIndex) == 0.f)
				prQueue.push(propagation);

			// bottom
			propagation.x = startEstimate.x;
			propagation.y = startEstimate.y + 1;
			const int bottomIndex = propagation.getLinearIdx(this->width);
			if (refV->confImg->at(bottomIndex) < minConfidence || refV->confImg->at(bottomIndex) == 0.f)
				prQueue.push(propagation);
		}
	}
}

void DMRecon::updateReferenceView(SingleView::Ptr &refV,
	const PatchOptimization &patch, const float confidence,
	const int pixelIdx, const bool keepViewIndicesPerPixel)
{
	// update reference view with depth, normal, dz, confidence and view IDs
	refV->depthImg->at(pixelIdx) = patch.getDepth();

	const math::Vec3f normal = patch.getNormal();
	refV->normalImg->at(pixelIdx, 0) = normal[0];
	refV->normalImg->at(pixelIdx, 1) = normal[1];
	refV->normalImg->at(pixelIdx, 2) = normal[2];

	refV->dzImg->at(pixelIdx, 0) = patch.getDzI();
	refV->dzImg->at(pixelIdx, 1) = patch.getDzJ();

	refV->confImg->at(pixelIdx) = confidence;

	if (keepViewIndicesPerPixel)
		storeViewIndices(refV, pixelIdx, patch);
}

void DMRecon::storeViewIndices(const SingleView::Ptr &refV, const int index, const PatchOptimization &patch)
{
	// store store reference view ID
	refV->viewIndicesImg->at(index, 0) = getRefViewNr();
	
	// store other view IDs from local selection
	IndexSet const& localViewIDs = patch.getLocalViewIDs();
	size_t nextViewIDIndex = 1;
	for (IndexSet::iterator it = localViewIDs.begin(); it != localViewIDs.end(); ++it)
		refV->viewIndicesImg->at(index, nextViewIDIndex++) = *it;
}

MVS_NAMESPACE_END
