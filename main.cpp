// Performance tester for linear regression

#include <stdio.h>

#include "LinearRegression_CUDA.cu"
//#include "LinearRegression_CUDA_NoSharedMemory.cu"
//#include "LinearRegression_cuBLAS.cu"
#include <chrono> // NOTE: official perf tests use Nsight not chrono

#include <vector>
#include <random>
#include <fstream>

// Data size
#define N 1000000 // Size of dataset
#define TRAIN_SIZE 800000 // Number of points to use for training
#define TEST_SIZE (N-TRAIN_SIZE) // Number of points to use for testing

// Random data parameters
#define M 25 // mx+b
#define B 50 // mx+b
#define MIN_X -10 // Minimum x value in dataset
#define MAX_X 10 // Maximum x value in dataset
#define NOISE_LEVEL 50 // How much noise
#define WRITE_FILE false // If you want to write data to file for debugging
#define FILENAME "cpp_data.csv" // Save data to this file

using namespace std;

// Function to generate a dataset
void generateDataset(float *x, float *y) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> disX(MIN_X, MAX_X);
    normal_distribution<> disNoise(0, NOISE_LEVEL);

    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(disX(gen));
        y[i] = M * x[i] + B + static_cast<float>(disNoise(gen));
    }

    // Write dataset to a file
    if (WRITE_FILE) {
        ofstream outFile(FILENAME);
        if (outFile.is_open()) {
            outFile << "x,y\n";
            for (int i = 0; i < N; ++i) {
                outFile << x[i] << "," << y[i] << "\n";
            }
            outFile.close();
        }
    }
}

int main() {
    chrono::time_point<chrono::system_clock> now = chrono::system_clock::now();

    // Dataset
    float x[N];
    float y[N];
    generateDataset(x,y);

    chrono::time_point<chrono::system_clock> mark1 = chrono::system_clock::now();
    LinearRegression_CUDA model(N, TRAIN_SIZE, TEST_SIZE, x, y); // UPDATE ME TO CHANGE MODELS
    chrono::time_point<chrono::system_clock> mark2 = chrono::system_clock::now();

    model.calculate_coefficients();
    chrono::time_point<chrono::system_clock> mark3 = chrono::system_clock::now();

    model.make_predictions();
    chrono::time_point<chrono::system_clock> mark4 = chrono::system_clock::now();

    model.calculate_mse();
    chrono::time_point<chrono::system_clock> mark5 = chrono::system_clock::now();

    auto m1 = chrono::duration_cast<chrono::nanoseconds>(mark1 - now).count();
    auto m2 = chrono::duration_cast<chrono::nanoseconds>(mark2 - mark1).count();
    auto m3 = chrono::duration_cast<chrono::nanoseconds>(mark3 - mark2).count();
    auto m4 = chrono::duration_cast<chrono::nanoseconds>(mark4 - mark3).count();
    auto m5 = chrono::duration_cast<chrono::nanoseconds>(mark5 - mark4).count();

    printf ("h data:\t %ld ns\n", m1);
    printf ("constructor:\t %ld ns\n", m2);
    printf ("coefficients:\t %ld ns\n", m3);
    printf ("predictions:\t %ld ns\n", m4);
    printf ("mse:\t %ld ns\n", m5);
}
