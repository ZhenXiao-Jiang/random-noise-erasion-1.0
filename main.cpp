#include <iostream>
#include "env pre.h"
#include "J_layers.h"

using namespace std;

const double TARGET_0 = 3.5;       //the bias demand 1(used to evaluate the model)
const double TARGET_1 = 4.5;       // the bias demand 2(used to evaluate the model)
const bool do_train = 1;         //whether to train the model and save it or to load the trained model saved in the file
const double learning_rate = 0.1;  //the learning rate of the model
const int update_interval = 1;		
const double momentum = 0.2;
const double decay_rate = 0.99;
const double min_lr = 0.001;

int main() {
	printf("falasdfasjkdfjkahdfkl");
	printf("adsfadf");
	J_liner_layer layer1(10, 20, 0.1, 1, 0.2, 0.99, 0.001);			 //the parameters are input_size, output_size, learning_rate, update_interval, momentum, decay_rate, min_learning_rate
	J_sigmoid_layer layer2(20);
	J_liner_layer layer3(20, 2, 0.1, 1, 0.2, 0.99, 0.001);
	Env env;
	vector<double> input,output,loss;

	//train
	if (do_train) {
		for (int episode = 0; episode < 200000; episode++) {			//the times of training are set here
			input.clear();
			env.reset();
			for (int i = 0; i < 5; i++) {
				env.step();
				vector<double> obs = env.observe();
				input.push_back(obs[0]);
				input.push_back(obs[1]);
			}
			output = layer3.forward(layer2.forward(layer1.forward(input)));
			loss = env.get_state();
			for (int i = 0; i < 2; i++) {
				loss[i] = output[i] - loss[i];
			}
			layer1.backward(layer2.backward(layer3.backward(loss)));
			if (episode % 10000 == 0) {
				cout << "Episode: " << episode / 10000 << ": " << input[8] - env.get_state()[0] << " " << input[9] - env.get_state()[1];
				cout << " Loss: " << loss[0] << " " << loss[1] << endl;
			}
		}

		//save
		layer1.save("layer1");
		layer3.save("layer3");
	}
	else {
		//load
		layer1.load("layer1");
		layer3.load("layer3");
	}
	
	//evaluation
	double max_0[2]{}, max_1[2]{};  //the maxium bias of the original observation and the output
	int loss_count_0[2]{};			 //the number of times the bias exceeds the target
	int loss_count_1[2]{};
	double avg_loss[2]{};          //the average bias
	for (int episode = 0; episode < 1000; episode++) {
		input.clear();
		env.reset();
		for (int i = 0; i < 5; i++) {
			env.step();
			vector<double> obs = env.observe();
			input.push_back(obs[0]);
			input.push_back(obs[1]);
		}
		output = layer3.forward(layer2.forward(layer1.forward(input)));
		loss = env.get_state();
		for (int i = 0; i < 2; i++) {
			input[i] = abs(input[i+8] - loss[i]);
			loss[i] = abs(output[i] - loss[i]);
			avg_loss[0] += input[i];
			avg_loss[1] += loss[i];
			if (input[i] > TARGET_0) {
				loss_count_0[0]++;
			}
			if (loss[i] > TARGET_0) {
				loss_count_0[1]++;
			}
			if (input[i] > TARGET_1) {
				loss_count_1[0]++;
			}
			if (loss[i] > TARGET_1) {
				loss_count_1[1]++;
			}
			if (input[i] > max_0[i]) {
				max_0[i] = input[i];
			}
			if (loss[i] > max_1[i]) {
				max_1[i] = loss[i];
			}
		}
	}
	cout << "Max obs bias: x: " << max_0[0] << " y: " << max_0[1] << " Loss count: " << TARGET_0 << ": " << loss_count_0[0] << "/2000 " << TARGET_1 << ": " << loss_count_1[0] << "/2000 average bias : " << avg_loss[0] / 2000.0 << endl;
	cout << "Max out bias: x: " << max_1[0] << " y: " << max_1[1] << " Loss count: " << TARGET_0 << ": " << loss_count_0[1] << "/2000 " << TARGET_1 << ": " << loss_count_1[1] << "/2000 average bias : " << avg_loss[1] / 2000.0 << endl;
}