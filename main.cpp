#include "Neural_network.h"

Neural_network model;

int main() {
	model.init();
	model.add_hidden_layer(10,20,0);
	model.add_output_layer(20, 2);
	model.train();
	model.evaluation();
}
