#include<iostream>
#include<math.h>
#include<vector>
#include<numeric>

using namespace std;

class LinearRegression
{

private:
	double m_b1{};
	double m_b0{};
public:

	LinearRegression() {

	}

	~LinearRegression() {

	}

	void fit(vector<double>x_train,vector<double>y_train)
	{
		auto x_mean = Mean(x_train);
		auto y_mean = Mean(y_train);
		double sum_x1 = 0;
		double sum_x2 = 0;
		for (int i = 0; i < x_train.size(); i++)
		{
			sum_x1 += ((x_train[i] - x_mean) * (y_train[i] - y_mean));
		}

		for (int i = 0; i < x_train.size(); i++)
		{
			sum_x2 += pow((x_train[i] - x_mean), 2);
		}

		m_b1 = sum_x1 / sum_x2;

		m_b0 = y_mean - (m_b1 * x_mean);

	}

	vector<double> predict(vector<double>x_test)
	{
		vector<double> pred_x{};

		for (int i = 0; i < x_test.size(); i++)
		{
			auto val = m_b0 + (m_b1 * x_test[i]);
			pred_x.push_back(val);
		}

		return pred_x;
	}

	double MSE(vector<double>y_test,vector<double>y_pred)
	{
		double mse_val{};
		auto sum = 0;
		for (int i = 0; i < y_test.size(); i++)
		{
			sum += pow((y_pred[i] - y_test[i]), 2);
		}
		mse_val = sum / (y_test.size() - 2);
		return mse_val;
	}
	double Mean(vector<double>arr_val)
	{
		return 1.0 * std::accumulate(arr_val.begin(), arr_val.end(), 0LL) / arr_val.size();
	}
};

int main()
{
	LinearRegression model;
	vector<double>x_train{ 1,2,3,4 };
	vector<double>y_train{ 1,2,3,4 };
	vector<double>x_test{ 5,6,7};
	vector<double>y_test{ 5,6,7};
	model.fit(x_train, y_train);
	auto y_pred = model.predict(x_test);
	auto MSE = model.MSE(y_test, y_pred);
	cout << MSE;
}