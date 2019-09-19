#include "network.h"

#include "helper.h"
#include "layer.h"

#include <iostream>
#include <iomanip>
#include <nvToolsExt.h>

using namespace cudl;

Network::Network()
{
	// nothing
}

Network::~Network()
{
	// destroy network
	for (auto layer: layers_)
		delete layer;

	// terminate CUDA context
	if (cuda_ != nullptr)
		delete cuda_;
}

void Network::add_layer(Layer *layer)
{
	layers_.push_back(layer);

	// tagging layer to stop gradient if it is the first layer
	if (layers_.size() == 1)
		layers_.at(0)->set_gradient_stop();
}

Blob<float> *Network::forward(Blob<float> *input)
{
	output_ = input;

	nvtxRangePushA("Forward");
	for (auto layer : layers_)
	{
		#if (DEBUG_FORWARD)
		std::cout << "[[Forward ]][[ " << std::setw(7) << layer->get_name() << " ]]\t(" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\t";
		#endif // DEBUG_FORWARD

		output_ = layer->forward(output_);

		#if (DEBUG_FORWARD)
		std::cout << "--> (" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")" << std::endl;
		checkCudaErrors(cudaDeviceSynchronize());

		#if (DEBUG_FORWARD > 1)
			output_->print("output", true);

			if (phase_ == inference)
				getchar();
		#endif
		#endif // DEBUG_FORWARD

		// TEST
		// checkCudaErrors(cudaDeviceSynchronize());
	}
	nvtxRangePop();

	return output_;
}

void Network::backward(Blob<float> *target)
{
	Blob<float> *gradient = target;

	if (phase_ == inference)
		return;

	nvtxRangePushA("Backward");
	// back propagation.. update weights internally.....
	for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
	{
		// getting back propagation status with gradient size

#if (DEBUG_BACKWARD)
		std::cout << "[[Backward]][[ " << std::setw(7) << (*layer)->get_name() << " ]]\t(" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\t";
#endif // DEBUG_BACKWARD

		gradient = (*layer)->backward(gradient);

#if (DEBUG_BACKWARD)
		// and the gradient result
		std::cout << "--> (" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")" << std::endl;
		checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_BACKWARD > 1)
		gradient->print((*layer)->get_name() + "::dx", true);
		getchar();
#endif
#endif // DEBUG_BACKWARD
	}
	nvtxRangePop();
}

void Network::update(float learning_rate)
{
	if (phase_ == inference)
		return;

#if (DEBUG_UPDATE)
	std::cout << "Start update.. lr = " << learning_rate << std::endl;
#endif

	nvtxRangePushA("Update");
	for (auto layer : layers_)
	{
		// if no parameters, then pass
		if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
			layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
			continue;

		layer->update_weights_biases(learning_rate);
	}
	nvtxRangePop();
}

int Network::write_file()
{
	std::cout << ".. store weights to the storage .." << std::endl;
	for (auto layer : layers_)
	{
		int err = layer->save_parameter();
		
		if (err != 0)
		{
			std::cout << "-> error code: " << err << std::endl;
			exit(err);
		}
	}

	return 0;
}

int Network::load_pretrain()
{
	for (auto layer : layers_)
	{
		layer->set_load_pretrain();
	}

	return 0;
}

// 1. initialize cuda resource container
// 2. register the resource container to all the layers
void Network::cuda()
{
	cuda_ = new CudaContext();

	std::cout << ".. model Configuration .." << std::endl;
	for (auto layer : layers_) {
		std::cout << "CUDA: " << layer->get_name() << std::endl;
		layer->set_cuda_context(cuda_);
	}
}

// 
void Network::train()
{
	phase_ = training;

	// unfreeze all layers
	for (auto layer : layers_)
	{
		layer->unfreeze();
	}
}

void Network::test()
{
	phase_ = inference;

	// freeze all layers
	for (auto layer : layers_)
	{
		layer->freeze();
	}
}

std::vector<Layer *> Network::layers()
{
	return layers_;
}

float Network::loss(Blob<float> *target)
{
	Layer *layer = layers_.back();
	return layer->get_loss(target);
}

int Network::get_accuracy(Blob<float> *target)
{
	Layer *layer = layers_.back();
	return layer->get_accuracy(target);
}


#if 0
Blob<float> *predict = this->output_;
	int batch_size = predict->n();
	int output_size = predict->c();

#if (DEBUG_ACCURACY)
	std::cout << "[[ ACCURACY ]]" << std::endl;
	predict->print("predict:", true);
	target->print("target:", true);
#endif // DEBUG_ACCURACY

	float* h_predict = predict->to(host);
	float* h_target  = target->to(host);
	cudaDeviceSynchronize();
	int result = 0;
	for (int b = 0; b < batch_size; b++)
	{
		int idx_predict = 0;
		int idx_target = 0;
		for (int j = 0; j < output_size; j++) {
			if (h_predict[b*output_size + j] > h_predict[idx_predict])
				idx_predict = j;
			// std::cout << "[" << j << "]" << h_target[b*output_size + j] << ", " << h_target[idx_predict] << std::endl;
			if (h_target[b*output_size + j] > h_target[idx_target])
				idx_target = j;
		}
		
#if (DEBUG_ACCURACY)
		std::cout << "predict:: " << idx_predict << ", target::" << idx_target << std::endl;
#endif // DEBUG_ACCURACY
		//std::cout << "p: " << idx_predict << ", y: " << idx_target << std::endl;

		if (idx_predict == idx_target)
			result++;
	}

#if (DEBUG_ACCURACY)
	getchar();
#endif // DEBUG_ACCURACY

#endif