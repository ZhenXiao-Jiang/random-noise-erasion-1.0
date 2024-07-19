#include <iostream>
#include <string>
#include "env pre.h"
#include "J_layers.h"

class Neural_network {
private:
	int _input_size, _output_size;

	int _layers_size = 0;
	std::vector<J_layer> hidden_layers;
	J_layer output_layer;

	double _learning_rate;  //the learning rate of the model
	int _update_interval;
	double _momentum;
	double _decay_rate;
	double _min_lr;

	Env env;

	std::vector<double> input, output, loss, INPUT, LOSS;

	double TARGET_0 = 3.5;
	double TARGET_1 = 4.5;

public:
	void init(int input_size = 10, int output_size = 2, double learning_rate = 0.1, int update_interval = 1, double momentum = 0.2, double decay_rate = 0.99, double min_lr = 0.001) {
		_input_size = input_size;
		_output_size = output_size;
		_learning_rate = learning_rate;
		_update_interval = update_interval;
		_momentum = momentum;
		_decay_rate = decay_rate;
		_min_lr = min_lr;
	}
	void add_hidden_layer(int input_size, int par_size, int activate_function_type) {
		hidden_layers.push_back(J_layer(input_size, activate_function_type, par_size, _learning_rate, _update_interval, _momentum, _decay_rate, _min_lr));
		_layers_size++;
	}
	void add_output_layer(int input_size, int par_size) {
		output_layer = J_layer(input_size, 4, par_size, _learning_rate, _update_interval, _momentum, _decay_rate, _min_lr);
	}
	void load() {
		for (int i = 0; i < _layers_size; i++) {
			hidden_layers[i].load("hidden_layer" + std::to_string(i));
		}
		output_layer.load("output_layer");
	}
	void save() {
		for (int i = 0; i < _layers_size; i++) {
			hidden_layers[i].save("hidden_layer" + std::to_string(i));
		}
		output_layer.save("output_layer");
	}
	void train(int episode_size = 200000, bool is_train = 0) {
		//if (is_train) load();
		for (int episode = 0; episode < episode_size; episode++) {			//the times of training are set here
			input.clear();
			env.reset();
			input = env.get_input(_input_size);
			INPUT = input;
			for (int i = 0; i < _layers_size; i++) {
				input = hidden_layers[i].forward(input);
			}
			input = output_layer.forward(input);
			output = input;
			loss = env.reward(output);
			LOSS = loss;
			loss = output_layer.backward(loss);
			for (int i = _layers_size - 1; i >= 0; i--) {
				loss = hidden_layers[i].backward(loss);
			}

			if (episode % 10000 == 0) {
				std::cout << "Episode: " << episode / 10000 << std::endl;
				std::cout << " Loss: " << LOSS[0] << " " << LOSS[1] << std::endl;
			}
		}
		//save();
	}
	void evaluation(int episode_size = 1000) {
		double max_0[2]{}, max_1[2]{};  //the maxium bias of the original observation and the output
		int loss_count_0[2]{};			 //the number of times the bias exceeds the target
		int loss_count_1[2]{};
		double avg_loss[2]{};          //the average bias
		for (int episode = 0; episode < 1000; episode++) {
			input.clear();
			env.reset();
			input = env.get_input(_input_size);
			INPUT = input;
			for (int i = 0; i < _layers_size; i++) {
				input = hidden_layers[i].forward(input);
			}
			input = output_layer.forward(input);
			output = input;
			loss = env.get_state();
			for (int i = 0; i < 2; i++) {
				INPUT[i] = abs(INPUT[i + 8] - loss[i]);
				loss[i] = abs(output[i] - loss[i]);
				avg_loss[0] += INPUT[i];
				avg_loss[1] += loss[i];
				if (INPUT[i] > TARGET_0) {
					loss_count_0[0]++;
				}
				if (loss[i] > TARGET_0) {
					loss_count_0[1]++;
				}
				if (INPUT[i] > TARGET_1) {
					loss_count_1[0]++;
				}
				if (loss[i] > TARGET_1) {
					loss_count_1[1]++;
				}
				if (INPUT[i] > max_0[i]) {
					max_0[i] = INPUT[i];
				}
				if (loss[i] > max_1[i]) {
					max_1[i] = loss[i];
				}
			}
		}
		std::cout << "Max obs bias: x: " << max_0[0] << " y: " << max_0[1] << " Loss count: " << TARGET_0 << ": " << loss_count_0[0] << "/" << std::to_string(2 * episode_size) + " " << TARGET_1 << ": " << loss_count_1[0] << "/" << std::to_string(2 * episode_size) << " average bias : " << avg_loss[0] / 2 / episode_size << std::endl;
		std::cout << "Max out bias: x: " << max_1[0] << " y: " << max_1[1] << " Loss count: " << TARGET_0 << ": " << loss_count_0[1] << "/" << std::to_string(2 * episode_size) + " " << TARGET_1 << ": " << loss_count_1[1] << "/" << std::to_string(2 * episode_size) << " average bias : " << avg_loss[1] / 2 / episode_size << std::endl;
	}
};