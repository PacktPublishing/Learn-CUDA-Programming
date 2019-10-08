#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include "helper.h"
#include "loss.h"
#include "layer.h"

namespace cudl {

typedef enum {
    training,
    inference
} WorkloadType;

class Network
{
    public:
    Network();
    ~Network();

    void add_layer(Layer *layer);

    Blob<float> *forward(Blob<float> *input);
    void backward(Blob<float> *input = nullptr);
    void update(float learning_rate = 0.02f);

    int load_pretrain();
    int write_file();

    float loss(Blob<float> *target);
    int get_accuracy(Blob<float> *target);

    void cuda();
    void train();
    void test();

    Blob<float> *output_;

    std::vector<Layer *> layers();


  private:
    std::vector<Layer *> layers_;

    CudaContext *cuda_ = nullptr;

    WorkloadType phase_ = inference;
};

} // namespace cudl


#endif // _NETWORK_H_