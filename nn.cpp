#include <stdio.h>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void construct_mlp()
{
    network<sequential> net;

    net << fc(2, 4) 
        << fc(4, 8)
        << fc(8, 2)
        << fc(2, 1);

    adagrad optimizer;

    std::vector<vec_t> train_labels;
    std::vector<vec_t> train_images;
    for (int i = -50; i < 50; i++)
        for (int j = -50; j < 50; j++)
        {
            float sum = (float)i + (float)j;
            train_labels.push_back({sum});
            train_images.push_back({(float)i, (float)j});
        }

    fprintf(stderr, "Training the neural net.\n");
    int epoch = 1;
    auto on_enumerate_epoch = [&]()
    {
        fprintf(stderr, "Finished epoch %d\n", epoch++);
    };
    auto on_batch = [&]() {};

    net.fit<mse>(
        optimizer, train_images, train_labels, 30, 10, []() {}, [&]()
        { fprintf(stderr, "Finished training epoch #%d\n", epoch++); });

    fprintf(stderr, "Finished training the neural net. Doing prediction of 4+5.\n");
    net.save("trained");
    vec_t output = net.predict({4, 5});
    fprintf(stderr, "Got output %f from the net.\n", output[0]);
}

// This file is responsible just for training the neural net.
int main()
{
    FILE *file;
    file = fopen("data/", "r");
    construct_mlp();
}
