#pragma once
#include <vector>
#include "J_random.h"

class Env {
private:
	std::vector<double> _state;
	double _v_x, _v_y;
public:
	Env(double x = 0, double y = 0, double v_x = 10, double v_y = 10) : _state({ x, y }), _v_x(v_x), _v_y(v_y) {}
	std::vector<double> get_state() { return _state; }
	std::vector<double> observe() { 
		std::vector<double> obs = { _state[0] + _v_x * double_random_dis_0(), _state[1] + _v_y * double_random_dis_0()};
		return obs; 
	}
	void step() {
		_state[0] += _v_x;
		_state[1] += _v_y;
	}
	void reset() {
		_state = { int_random()/10.0, int_random() / 10.0 };
	}
};