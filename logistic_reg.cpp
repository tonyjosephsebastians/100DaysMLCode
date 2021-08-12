#include<iostream>
#include<vector>
#include<math.h>
#include<string>
#include<fstream>
using namespace std;


class LogisticRegression
{
	double activation(double z);
	void update_weight(std::vector<double> x_train, double y_train, double predicted_value);
public:
	void fit(std::vector<vector<double>>x_train, std::vector<double>y_train);
	double predict(std::vector<double>x_test);
	double accuracy(std::vector<vector<double>> x_test, std::vector<double> y_test);
	//double accuracy(std::vector<vector<double>>x_test);
	vector<double>m_weight{};
	double m_learningRate = 0.001;
	long m_epoch = 10;
	double e = 2.71828;

};

int main()
{

	vector<vector<double>> x_train;
	vector<double>y_train;
	x_train.clear();
	y_train.clear();

	system("clear"); //for windows use system("cls")

	string line;
	long i, j; //general purpose counter

	ifstream infile;
	string file = "C://Users//TONY//source//repos//logistic_regression//iris.csv";
	infile.open(file.c_str());
	if (infile.is_open())
	{
		cout << "file opened";
	}

	while (getline(infile, line)) {
		//to skip the first two entries that represent
		//Id and the comma(,) following it
		i = 0;
		while (line[i] != ',') {
			i++;
		}
		i++;

		int token; //the values to be read from file
		vector<double> inputRow;
		double output;

		inputRow.clear();

		for (token = 0; token < 4; token++) {
			double value;
			string val = "";
			while (line[i] != ',') {
				val += line[i];
				i++;
			}
			i++;
			value = stod(val);
			inputRow.push_back(value);
		}
		i++;

		string outputStr = "";

		outputStr = line[line.size() - 1];
		output = stod(outputStr);

		y_train.push_back(output);
		x_train.push_back(inputRow);
	}
	LogisticRegression lr;
	lr.fit(x_train, y_train);
	double acc = lr.accuracy(x_train, y_train);
	cout << acc;

	vector<double>test{};
	test.push_back(1);
	test.push_back(2);
	test.push_back(3);
	test.push_back(5);

	int pred = lr.predict(test);
	cout << pred;
	return 0;
	}



void LogisticRegression::fit(std::vector<vector<double>>x_train, std::vector<double>y_train)
{

	for (int k = 0; k < x_train[0].size(); k++)
	{
		m_weight.push_back(0.2);
	}

	if (x_train.size() != y_train.size())
	{
		printf("Error");
	}
	while (m_epoch--)
	{
		for (int i = 0; i < x_train.size(); i++)
		{
			double z{};
			double predicted_value{};
			for (int j = 0; j < x_train[0].size(); j++)
			{
				z += m_weight[j] * x_train[i][j];
			}
			predicted_value = activation(z);
			update_weight(x_train[i], y_train[i], predicted_value);
		}
	}
}

double LogisticRegression::activation(double z)
{
	return 1 / (1 + pow(e, (-1 * z)));
}

void LogisticRegression::update_weight(std::vector<double>x_train,double y_train, double predicted_value)
{
	double gradient{};
	for (int i = 0; i < x_train.size(); i++)
	{
		gradient = (predicted_value - y_train) * x_train[i];
		m_weight[i] = m_weight[i] - m_learningRate * gradient;
	}
}

double LogisticRegression::predict(std::vector<double>x_test)
{
	double pred{};
	double z{};
	for (int i = 0; i < x_test.size(); i++)
	{

		z += m_weight[i] * x_test[i];
	}
	pred = activation(z);
	
	if (pred < 0.5)
	{
		return 0;
	}

	else
	{
		return 1;
	}
}
double LogisticRegression::accuracy(std::vector<vector<double>>x_test, std::vector<double>y_test)
{

	int total_correct{};
	double accuracy{};
	for (int i = 0; i < x_test.size(); i++)
	{
		double predicted_value{};
		double z{};
		for (int j = 0; j < x_test[0].size(); j++)
		{
			z += m_weight[j] * x_test[i][j];
		}
		predicted_value = activation(z);

		if (predicted_value < 0.5)
		{
			predicted_value = 0;
		}

		else
		{
			predicted_value = 1;
		}
		if (predicted_value == y_test[i])
		{
			total_correct += 1;
		}

	}

	accuracy = (total_correct * 100 / x_test.size());
	return accuracy;
}