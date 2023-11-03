#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <regex>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

struct Args
{
    int batch_size = -1;
    int num_epochs = -1;
    string testnet;
    string data_file;
};

bool starts_with(const string &str, const string &prefix)
{
    regex re{"^" + prefix + ".*"};
    return regex_match(str, re);
}

bool ParseArgs(Args &args, int argc, char **argv)
{
    map<string, string> arg_map;
    for (int i = 1; i < argc; i++)
    {
        string arg(argv[i]);
        if (starts_with(arg, "--"))
        {
            string name = arg.substr(2);
            string value = argv[i + 1];
            arg_map[name] = value;
            i++;
        }
    }

    if (arg_map.count("batch-size"))
    {
        args.batch_size = stoi(arg_map["batch-size"]);
    }

    if (arg_map.count("num-epochs"))
    {
        args.num_epochs = stoi(arg_map["num-epochs"]);
    }

    if (arg_map.count("testnet"))
    {
        args.testnet = arg_map["testnet"];
    }

    if (arg_map.count("data-file"))
    {
        args.data_file = arg_map["data-file"];
    }

    return true;
}

float str_to_float(const std::string &s)
{
    try
    {
        return std::stof(s);
    }
    catch (...)
    {
        return 0.0;
    }
}

int main(int argc, char **argv)
{
    Args args;
    // Set sane defaults.
    args.batch_size = 10;
    args.num_epochs = 5;
    args.data_file = "./data.csv";
    args.testnet = "testnet";

    if (!ParseArgs(args, argc, argv))
    {
        cout << "Usage: " << argv[0] << " --batch-size <number> --num-epochs <number> --testnet <filename> --data-file data file as csv format." << endl;
        return 1;
    }

    vector<vec_t> boards;
    vector<vec_t> wins;

    std::ifstream datafile(args.data_file);

    std::string line;
    while (std::getline(datafile, line))
    {
        std::istringstream stream(line);
        vec_t board;
        vec_t win;

        string word;
        for (int i = 0; i < 32; i++)
        {
            stream >> word;
            board.push_back(str_to_float(word));
        }

        boards.push_back(board);
        stream >> word;

        win.push_back(static_cast<float_t>(std::stoi(word)));
        wins.push_back(win);
    }

    network<sequential> net;
    try
    {
        net.load(args.testnet);
        fprintf(stderr, "Found an old testnet. Using it.\n");
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Could not find a testnet. Creating one.\n");
        core::backend_t backend_type = core::default_engine();

        net << fully_connected_layer(32, 32*32) << relu()
            << convolutional_layer(32, 32, 5, 1,
                                   6)
            << tanh_layer(28, 28, 6)
            << average_pooling_layer(28, 28, 6,
                                     2) // S2, 6@28x28-in, 6@14x14-out
            << tanh_layer(14, 14, 6)
            << convolutional_layer(14, 14, 5, 6,
                                   16)
            << tanh_layer(10, 10, 16)
            << average_pooling_layer(10, 10, 16,
                                     2) // S4, 16@10x10-in, 16@5x5-out
            << tanh_layer(5, 5, 16)
            << convolutional_layer(5, 5, 5, 16,
                                   120)
            << tanh_layer(1, 1, 120)
            << fully_connected_layer(120, 2, true, // F6, 120-in, 10-out
                                     backend_type)
            << fully_connected_layer(2, 1);
    }

    cerr << "Got " << boards.size() << " boards and " << wins.size() << " labels for training. " << endl;

    // Run the net on the new data.
    adagrad optimizer;
    int epoch = 0;

    net.fit<mse>(
        optimizer, boards, wins, args.batch_size, args.num_epochs, []() {}, [&]()
        { 
            std::cout << "Finished training epoch: " << epoch++; 
            auto loss = net.get_loss<mse>(boards, wins); cout << " loss: " << loss << endl; });

    net.save(args.testnet);
}
