#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

int main() {
  network<sequential> net;
  net << fully_connected_layer(64, 8 * 8);
  net << relu()
    << fully_connected_layer(32, 64)
    << relu()
    << fully_connected_layer(16, 32);
  net << fully_connected_layer(3, 16) << softmax();

  adam optimizer;

  std::vector<vec_t> input_data; // Input data (board state)
  std::vector<label_t> target_data; // Target data (win, lose, draw)

  // Custom board state
  input_data.push_back({ 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0 });

  // Corresponding target
  target_data.push_back(1);

/*
    std::ifstream file("custom_dataset.txt");

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the dataset file." << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        vec_t board_state; //  store the board state

        int value;
        while (iss >> value) {
            if (iss.eof()) {
                // The last value on the line is the target
                target_data.push_back(static_cast<label_t>(value));
            } else {
                // Values before the last one are part of the board state
                board_state.push_back(static_cast<float_t>(value));
            }
        }

        input_data.push_back(board_state);
    }
*/

  int num_epochs = 100;
  int batch_size = 32;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<int> indices(input_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(0));

    for (size_t i = 0; i < indices.size(); i += batch_size) {
      int end = std::min(static_cast<int>(i + batch_size), static_cast<int>(indices.size()));
      std::vector<vec_t> batch_input;
      std::vector<label_t> batch_target;

      for (int j = i; j < end; j++) {
        batch_input.push_back(input_data[indices[j]]);
        batch_target.push_back(target_data[indices[j]]);
      }

      net.fit<cross_entropy>(optimizer, batch_input, batch_target);

    std::cout << "Epoch: " << epoch + 1 << " Batch: " << i / batch_size + 1 << " Loss: " << net.get_loss<cross_entropy>(batch_input, batch_target) << std::endl;
    }
  }

  net.save("checkers_mlp_model");
  return 0;
}
