#pragma once
#include <vector>
#include <fstream>

class J_liner_layer {
private:
	int _input_size, _output_size, _update_interval, _update_count;
	double _learning_rate, _decay_rate, _min_learning_rate, _momentum;
	std::vector<std::vector<double>> _w, _update_w;
	std::vector<double> _b, _update_b, _grad_w;
public:
	J_liner_layer(int input_size = 1, int output_size = 4, double learning_rate = 0.001, int update_interval = 1, double momentum = 0, double decay_rate = -1, double min_learning_rate = -1, bool xavier_ini = 0) {
		_input_size = input_size;
		_output_size = output_size;
		_learning_rate = learning_rate;
		_update_interval = update_interval;
		_update_count = 0;
		_momentum = momentum;
		_decay_rate = decay_rate;
		_min_learning_rate = min_learning_rate > 0 ? min_learning_rate : learning_rate/100.0;
		_w.resize(_output_size);
		_grad_w.resize(_input_size);
		_update_w.resize(_output_size);
		_b.resize(_output_size);
		_update_b.resize(_output_size);
		for (int i = 0; i < _output_size; i++) {
			_w[i].resize(_input_size);
			_update_w[i].resize(_input_size);
			for (int j = 0; j < _input_size; j++) {
				_w[i][j] = (rand() % 1000) / 1000.0 - 0.5;
				_update_w[i][j] = 0;
			}
		}
		for (int i = 0; i < _output_size; i++) {
			_b[i] = (rand() % 1000) / 1000.0 - 0.5;
			_update_b[i] = 0;
		}
		if (xavier_ini) {
			_w = xavier_init(_input_size, _output_size);
		}
	};

	std::vector<double> forward(std::vector<double> input) {
		if(input.size() != _input_size) {
			std::cout << "input size error" << std::endl;
			exit(1);
		}
		std::vector<double> output(_output_size);
		for (int i = 0; i < _output_size; i++) {
			output[i] = 0;
			for (int j = 0; j < _input_size; j++) {
				output[i] += input[j] * _w[i][j];
			}
			output[i] += _b[i];
		}
		for (int i = 0; i < _input_size; i++) {
			_grad_w[i] = input[i];
		}
		return output;
	};

	std::vector<double> backward(std::vector<double> loss) {
		if(loss.size() != _output_size) {
			std::cout << "loss size error" << std::endl;
			exit(1);
		}
		_update_count++;
		std::vector<double> input(_input_size);
		for (int i = 0; i < _output_size; i++) {
			for (int j = 0; j < _input_size; j++) {
				_update_w[i][j] += loss[i] * _grad_w[j];
			}
			_update_b[i] += loss[i];
			for (int j = 0; j < _input_size; j++) {
				input[j] += loss[i] * _w[i][j];
			}
		}
		if (_update_count >= _update_interval) {
			for (int i = 0; i < _output_size; i++) {
				for (int j = 0; j < _input_size; j++) {
					_update_w[i][j] /= _update_count;
					_w[i][j] -= _learning_rate * _update_w[i][j];
					_update_w[i][j] *= _momentum;
				}
				_update_b[i] /= _update_count;
				_b[i] -= _learning_rate * _update_b[i];
				_update_b[i] *= _momentum;
			}
			if (_decay_rate > 0) {
				_learning_rate = std::max(_learning_rate * _decay_rate, _min_learning_rate);
			}
			_update_count = 0;
		}
		return input;
	};

	void save(std::string label = "J_liner_layer") {
		std::ofstream file;
		file.open(label + ".txt");
		file << 37 << " J_liner_layer" << std::endl;
		file << _input_size << " " << _output_size << " " << _learning_rate << " " << _update_interval << " " << _momentum << " " << _decay_rate << " " << _min_learning_rate << std::endl;
		for (int i = 0; i < _output_size; i++) {
			for (int j = 0; j < _input_size; j++) {
				file << _w[i][j] << " ";
			}
			file << _b[i] << " " << std::endl;
		}
		file.close();
	};

	void load(std::string label = "J_liner_layer") {
		std::ifstream file;
		file.open(label + ".txt");
		int id;
		std::string name;
		file >> id >> name;
		if(id != 37 || name != "J_liner_layer") {
			std::cout << "load error" << std::endl;
			exit(1);
		}
		file >> _input_size >> _output_size >> _learning_rate >> _update_interval >> _momentum >> _decay_rate >> _min_learning_rate;
		_w.resize(_output_size);
		_grad_w.resize(_input_size);
		_update_w.resize(_output_size);
		_b.resize(_output_size);
		_update_b.resize(_output_size);
		for (int i = 0; i < _output_size; i++) {
			_w[i].resize(_input_size);
			_update_w[i].resize(_input_size);
			for (int j = 0; j < _input_size; j++) {
				file >> _w[i][j];
				_update_w[i][j] = 0;
			}
			file >> _b[i];
			_update_b[i] = 0;
		}
		file.close();
	};
};

class J_relu_layer {
private:
	int _size;
	std::vector<double> _grad;
public:
	J_relu_layer(int size = 1) {
		_size = size;
		_grad.resize(_size);
	};

	std::vector<double> forward(std::vector<double> input) {
		if(input.size() != _size) {
			std::cout << "input size error" << std::endl;
			exit(1);
		}
		std::vector<double> output(_size);
		for (int i = 0; i < _size; i++) {
			if(input[i] > 0) {
				output[i] = input[i];
				_grad[i] = 1;
			} else {
				output[i] = 0;
				_grad[i] = 0;
			}
		}
		return output;
	};

	std::vector<double> backward(std::vector<double> loss) {
		if(loss.size() != _size) {
			std::cout << "loss size error" << std::endl;
			exit(1);
		}
		std::vector<double> input(_size);
		for (int i = 0; i < _size; i++) {
			input[i] = loss[i] * _grad[i];
		}
		return input;
	};
};

class J_leaky_relu_layer {
private:
	int _size;
	std::vector<double> _grad;
	double _alpha;
public:
	J_leaky_relu_layer(int size = 1, double alpha = 0.01) {
		_size = size;
		_alpha = alpha;
		_grad.resize(_size);
	};

	std::vector<double> forward(std::vector<double> input) {
		if(input.size() != _size) {
			std::cout << "input size error" << std::endl;
			exit(1);
		}
		std::vector<double> output(_size);
		for (int i = 0; i < _size; i++) {
			if(input[i] > 0) {
				output[i] = input[i];
				_grad[i] = 1;
			} else {
				output[i] = _alpha * input[i];
				_grad[i] = _alpha;
			}
		}
		return output;
	};

	std::vector<double> backward(std::vector<double> loss) {
		if(loss.size() != _size) {
			std::cout << "loss size error" << std::endl;
			exit(1);
		}
		std::vector<double> input(_size);
		for (int i = 0; i < _size; i++) {
			input[i] = loss[i] * _grad[i];
		}
		return input;
	};

};

class J_sigmoid_layer {
private:
	int _size;
	std::vector<double> _grad;
public:
	J_sigmoid_layer(int size = 1) {
		_size = size;
		_grad.resize(_size);
	};

	std::vector<double> forward(std::vector<double> input) {
		if(input.size() != _size) {
			std::cout << "input size error" << std::endl;
			exit(1);
		}
		std::vector<double> output(_size);
		for (int i = 0; i < _size; i++) {
			output[i] = 1.0 / (1.0 + exp(-input[i]));
			_grad[i] = output[i] * (1 - output[i]);
		}
		return output;
	};

	std::vector<double> backward(std::vector<double> loss) {
		if(loss.size() != _size) {
			std::cout << "loss size error" << std::endl;
			exit(1);
		}
		std::vector<double> input(_size);
		for (int i = 0; i < _size; i++) {
			input[i] = loss[i] * _grad[i];
		}
		return input;
	};
};

class J_tanh_layer {
private:
	int _size;
	std::vector<double> _grad;
public:
	J_tanh_layer(int size = 1) {
		_size = size;
		_grad.resize(_size);
	};

	std::vector<double> forward(std::vector<double> input) {
		if(input.size() != _size) {
			std::cout << "input size error" << std::endl;
			exit(1);
		}
		std::vector<double> output(_size);
		for (int i = 0; i < _size; i++) {
			output[i] = tanh(input[i]);
			_grad[i] = 1 - output[i] * output[i];
		}
		return output;
	};

	std::vector<double> backward(std::vector<double> loss) {
		if(loss.size() != _size) {
			std::cout << "loss size error" << std::endl;
			exit(1);
		}
		std::vector<double> input(_size);
		for (int i = 0; i < _size; i++) {
			input[i] = loss[i] * _grad[i];
		}
		return input;
	};
};

class J_softmax_layer {
private:
	int _size;
	std::vector<std::vector<double>> _grad;
public:
	J_softmax_layer(int size = 1) {
		_size = size;
		_grad.resize(_size);
		for (int i = 0; i < _size; i++) {
			_grad[i].resize(_size);
		}
	};

	std::vector<double> forward(std::vector<double> input) {
		if(input.size() != _size) {
			std::cout << "input size error" << std::endl;
			exit(1);
		}
		std::vector<double> output(_size);
		double sum = 0;
		for (int i = 0; i < _size; i++) {
			output[i] = exp(input[i]);
			sum += output[i];
		}
		for (int i = 0; i < _size; i++) {
			output[i] /= sum;
		}
		for (int i = 0; i < _size; i++) {
			for (int j = 0; j < _size; j++) {
				if(i == j) {
					_grad[i][j] = output[i] * (1 - output[i]);
				} else {
					_grad[i][j] = -output[i] * output[j];
				}
			}
		}
		return output;
	};

	std::vector<double> backward(std::vector<double> loss) {
		if(loss.size() != _size) {
			std::cout << "loss size error" << std::endl;
			exit(1);
		}
		std::vector<double> input(_size);
		for(int i = 0; i < _size; i++) {
			input[i] = 0;
			for(int j = 0; j < _size; j++) {
				input[i] += loss[j] * _grad[j][i];
			}
		}
		return input;
	};
};

class J_layer {
private:
	int _par_size;

	int _input_size;

	int _activate_function_type;

	J_liner_layer _liner;
	J_relu_layer _relu;
	J_leaky_relu_layer _leaky_relu;
	J_sigmoid_layer _sigmoid;
	J_tanh_layer _tanh;

public:
	J_layer(int input_size = 10, int activate_function_type = 4, int par_size = 20, double learning_rate = 0.1, int update_interval = 1, double momentum = 0.2, double decay_rate = 0.99, double min_learning_rate = 0.001) {
		_input_size = input_size;
		_par_size = par_size;
		_activate_function_type = activate_function_type;
		_liner = J_liner_layer(_input_size, _par_size, learning_rate, update_interval, momentum, decay_rate, min_learning_rate);
		switch (_activate_function_type) {
		case 0:
			_relu = J_relu_layer(_par_size);
			break;
		case 1:
			_leaky_relu = J_leaky_relu_layer(_par_size);
			break;
		case 2:
			_sigmoid = J_sigmoid_layer(_par_size);
			break;
		case 3:
			_tanh = J_tanh_layer(_par_size);
			break;
		case 4:
			break;
		default:
			std::cout << "activate function type error" << std::endl;
			exit(1);
		}
	};
	std::vector<double> forward(std::vector<double> input) {
		std::vector<double> ret = _liner.forward(input);
		switch (_activate_function_type) {
		case 0:
			ret = _relu.forward(ret);
			break;
		case 1:
			ret = _leaky_relu.forward(ret);
			break;
		case 2:
			ret = _sigmoid.forward(ret);
			break;
		case 3:
			ret = _tanh.forward(ret);
			break;
		case 4:
			break;
		default:
			std::cout << "activate function type error" << std::endl;
			exit(1);
		}
		return ret;
	};
	std::vector<double> backward(std::vector<double> input) {
		std::vector<double> ret = input;
		switch (_activate_function_type) {
		case 0:
			ret = _relu.backward(input);
			break;
		case 1:
			ret = _leaky_relu.backward(input);
			break;
		case 2:
			ret = _sigmoid.backward(input);
			break;
		case 3:
			ret = _tanh.backward(input);
			break;
		case 4:
			break;
		default:
			std::cout << "activate function type error" << std::endl;
			exit(1);
		}
		ret = _liner.backward(ret);
		return ret;
	};
	void save(std::string label = "J_layer") {
		_liner.load(label);
	}
	void load(std::string label = "J_layer") {
		_liner.load(label);
	}	
};