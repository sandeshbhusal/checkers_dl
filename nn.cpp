#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
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
    string data_dir;
};


bool starts_with(const string& str, const string& prefix) {
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

    if (arg_map.count("data-dir"))
    {
        args.data_dir = arg_map["data-dir"];
    }

    return true;
}

void enumerateFiles(const std::string &path, vector<string> &list_ptr)
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            if (ent->d_type == DT_REG)
            {
                string fullpath = path + "/" + ent->d_name;
                list_ptr.push_back(fullpath);
            }
        }
        closedir(dir);
    }
    else
    {
        std::cerr << "Could not open directory." << std::endl;
    }
}

int main(int argc, char** argv)
{
    Args args;
    // Set sane defaults.
    args.batch_size = 10;
    args.num_epochs = 5;
    args.data_dir = "./data/";
    args.testnet = "testnet";

    if (!ParseArgs(args, argc, argv))
    {
        cout << "Usage: " << argv[0] << " --batch-size <number> --num-epochs <number> --testnet <filename> --data-dir <string/path> " << endl;
        return 1;
    }

    vector<vec_t> boards;
    vector<vec_t> wins;

    vector<string> datafiles;
    enumerateFiles(args.data_dir, datafiles);

    for (auto file : datafiles)
    {
        float_t winner = 0.0;
        int num_boards = 0;

        ifstream DataFile(file);
        string line;
        while (getline(DataFile, line))
        {
            if (line.size() == 1)
            {
                // set winner here and break out of the loop.
                winner = static_cast<float_t>(std::stoi(line));
                break;
            }
            else
            {
                num_boards += 1;
                std::istringstream stream(line);
                string sub;
                vec_t board;
                while (stream >> sub)
                    board.push_back(static_cast<float_t>(std::stoi(sub)));

                boards.push_back(board);
            }
        }

        for (int i = 0; i < num_boards; i++)
        {
            wins.push_back({winner});
        }
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
        net << fully_connected_layer(32, 32) << sigmoid()
            << fully_connected_layer(32, 28) << sigmoid()
            << fully_connected_layer(28, 24) << sigmoid()
            << fully_connected_layer(24, 12) << sigmoid()
            << fully_connected_layer(12, 1) << sigmoid();
    }

    // Run the net on the new data.
    adagrad optimizer;
    int epoch = 0;
    net.fit<mse>(
        optimizer, boards, wins, args.batch_size, args.num_epochs, []() {}, [&]()
        { std::cout << "Finished training epoch: " << epoch++ << std::endl; });

    // Let's see if the network can predict the board any better now (should not be able to).
    // You need to provide a vector for prediction, not a single value.
    fprintf(stderr, "expected: %f, got: %f", wins[0][0], net.predict(boards[0])[0]);
    // Alsp calculate and print the loss.
    auto loss = net.get_loss<mse>(boards, wins);
    cout << "loss: " << loss << endl;
    net.save(args.testnet);
}
