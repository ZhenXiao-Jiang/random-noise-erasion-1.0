#pragma once
#include <random>
int int_random(int a = 0, int b = 100) {
	if (a > b) {
		int temp = a;
		a = b;
		b = temp;
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(a, b);
	return dis(gen);
}

double double_random_dis_0(double mean = 0, double stddev = 1, double max = 5) {
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> distribution(mean, stddev);
	while (1) {
		double num = distribution(gen);
		if (num < max && num > -max) {
			return num;
		}
	}
}