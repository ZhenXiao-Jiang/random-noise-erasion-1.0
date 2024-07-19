#include <iostream>
#include "env pre.h"
#include "J_layers.h"

using namespace std;

int main() {
	J_liner_layer layer1(10, 20);
	J_sigmoid_layer layer2(20);
	J_liner_layer layer3(20, 2);
	Env env;
	vector<double> input,output,loss;
	for(int episode = 0 ; episode < 1000000; episode++){
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
			cout << "Episode: " << episode/10000 << ": " << input[8] - env.get_state()[0] << " " << input[9] - env.get_state()[1];
			cout << " Loss: " << loss[0] << " " << loss[1] << endl;
		}
	}
}