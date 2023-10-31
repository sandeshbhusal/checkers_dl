#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void enumerateFiles(const std::string& path, vector<string> &list_ptr) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            if (ent->d_type == DT_REG) {
                string fullpath = path + "/" + ent->d_name;
                list_ptr.push_back(fullpath);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open directory." << std::endl;
    }
}

int main()
{
    vector<vec_t> boards;
    vector<vec_t> wins;

    vector<string> datafiles;
    enumerateFiles("./data/", datafiles);

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
        net.load("testnet");
        fprintf(stderr, "Found an old testnet. Using it.\n");
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Could not find a testnet. Creating one.\n");
        net << fully_connected_layer(32, 32) << relu()
            << fully_connected_layer(32, 28) << relu()
            << fully_connected_layer(28, 24) << relu()
            << fully_connected_layer(24, 12) << relu()
            << fully_connected_layer(12, 1) << sigmoid();
    }

    // Run the net on the new data.
    adagrad optimizer;
    int epoch = 0;
    net.fit<mse>(
        optimizer, boards, wins, 30, 10, []() {}, [&]()
        { std::cout << "Finished training epoch: " << epoch++ << std::endl; });

    // Let's see if the network can predict the board any better now (should not be able to).
    // You need to provide a vector for prediction, not a single value.
    fprintf(stderr, "expected: %f, got: %f", wins[0][0], net.predict(boards[0])[0]);
    net.save("testnet");
}
