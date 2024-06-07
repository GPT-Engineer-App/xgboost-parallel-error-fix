# xgboost-parallel-error-fix

J'ai l'erreur suivante : 
g++ -std=c++11 -pthread -o XGBoost_Parallele XGBoost_Parallele.cpp
XGBoost_Parallele.cpp: In member function ‘void DecisionTree::find_best_split(const std::vector<std::vector<double> >&, const std::vector<double>&, int&, double&)’:
XGBoost_Parallele.cpp:62:31: error: ‘numeric_limits’ was not declared in this scope
   62 |         double min_impurity = numeric_limits<double>::max();
      |                               ^~~~~~~~~~~~~~
XGBoost_Parallele.cpp:62:46: error: expected primary-expression before ‘double’
   62 |         double min_impurity = numeric_limits<double>::max();
      |                                              ^~~~~~
make: *** [Makefile:14 : XGBoost_Parallele] Erreur 1

lorsque je lance un make sur le code source suivant : 

/* MEGNA MFOUAKIE Ibrahim 18Z2256 Master 2 Informatique Option Data Science*/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <fstream>
#include <pthread.h>

using namespace std;

struct TreeNode {
    double value;
    int feature_index;
    double threshold;
    TreeNode* left;
    TreeNode* right;
    TreeNode(double val) : value(val), feature_index(-1), threshold(0.0), left(nullptr), right(nullptr) {}
};

class DecisionTree {
public:
    DecisionTree(int max_depth) : max_depth(max_depth) {}
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        root = build_tree(X, y, 0);
    }
    double predict(const vector<double>& x) const {
        return predict_recursive(root, x);
    }
private:
    TreeNode* root;
    int max_depth;
    TreeNode* build_tree(const vector<vector<double>>& X, const vector<double>& y, int depth) {
        if (depth == max_depth || X.empty()) {
            double leaf_value = calculate_leaf_value(y);
            return new TreeNode(leaf_value);
        }
        int best_feature;
        double best_threshold;
        find_best_split(X, y, best_feature, best_threshold);
        TreeNode* node = new TreeNode(0.0);
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        vector<vector<double>> left_X, right_X;
        vector<double> left_y, right_y;
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][best_feature] < best_threshold) {
                left_X.push_back(X[i]);
                left_y.push_back(y[i]);
            } else {
                right_X.push_back(X[i]);
                right_y.push_back(y[i]);
            }
        }
        node->left = build_tree(left_X, left_y, depth + 1);
        node->right = build_tree(right_X, right_y, depth + 1);
        return node;
    }
    void find_best_split(const vector<vector<double>>& X, const vector<double>& y, int& best_feature, double& best_threshold) {
        double min_impurity = numeric_limits<double>::max();
        for (int feature_idx = 0; feature_idx < X[0].size(); ++feature_idx) {
            vector<double> feature_values(X.size());
            for (size_t i = 0; i < X.size(); ++i) {
                feature_values[i] = X[i][feature_idx];
            }
            sort(feature_values.begin(), feature_values.end());
            for (size_t i = 0; i < feature_values.size() - 1; ++i) {
                double threshold = (feature_values[i] + feature_values[i + 1]) / 2;
                double impurity = calculate_impurity(X, y, feature_idx, threshold);
                if (impurity < min_impurity) {
                    min_impurity = impurity;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }
    }
    double calculate_impurity(const vector<vector<double>>& X, const vector<double>& y, int feature_idx, double threshold) {
        vector<double> left_y, right_y;
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][feature_idx] < threshold) {
                left_y.push_back(y[i]);
            } else {
                right_y.push_back(y[i]);
            }
        }
        double left_mean = accumulate(left_y.begin(), left_y.end(), 0.0) / left_y.size();
        double right_mean = accumulate(right_y.begin(), right_y.end(), 0.0) / right_y.size();
        double left_impurity = 0.0, right_impurity = 0.0;
        for (double val : left_y) {
            left_impurity += pow(val - left_mean, 2);
        }
        for (double val : right_y) {
            right_impurity += pow(val - right_mean, 2);
        }
        return (left_impurity + right_impurity) / y.size();
    }

    double calculate_leaf_value(const vector<double>& y) {
        return accumulate(y.begin(), y.end(), 0.0) / y.size();
    }
    double predict_recursive(TreeNode* node, const vector<double>& x) const {
        if (!node->left && !node->right) {
            return node->value;
        }
        if (x[node->feature_index] < node->threshold) {
            return predict_recursive(node->left, x);
        } else {
            return predict_recursive(node->right, x);
        }
    }
};

struct ThreadData {
    DecisionTree* tree;
    const vector<vector<double>>* X;
    const vector<double>* y;
    vector<double>* residuals;
    double learning_rate;
    int start;
    int end;
};

void* fit_tree(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->tree->fit(*data->X, *data->y);
    vector<double> predictions(data->X->size());
    for (size_t i = 0; i < data->X->size(); ++i) {
        predictions[i] = data->tree->predict((*data->X)[i]);
    }
    for (int i = data->start; i < data->end; ++i) {
        (*data->residuals)[i] -= data->learning_rate * predictions[i];
    }
    pthread_exit(nullptr);
}

class XGBoost {
public:
    XGBoost(int n_estimators, int max_depth, double learning_rate) : n_estimators(n_estimators), max_depth(max_depth), learning_rate(learning_rate) {}
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        vector<double> residuals = y;
        train_loss.clear();
        pthread_t threads[n_estimators];
        ThreadData thread_data[n_estimators];
        for (int i = 0; i < n_estimators; ++i) {
            unique_ptr<DecisionTree> tree(new DecisionTree(max_depth));
            thread_data[i] = {tree.get(), &X, &residuals, &residuals, learning_rate, 0, (int)X.size()};
            pthread_create(&threads[i], nullptr, fit_tree, (void*)&thread_data[i]);
            trees.push_back(move(tree));
        }
        for (int i = 0; i < n_estimators; ++i) {
            pthread_join(threads[i], nullptr);
        }
        // Calculate and store training loss
        double loss = calculate_loss(y, residuals);
        train_loss.push_back(loss);
        save_train_loss_to_file("train_loss.txt");
        save_visualization_to_file("train_loss.plt");
    }
    vector<double> predict(const vector<vector<double>>& X) {
        vector<double> predictions(X.size(), 0.0);
        for (const auto& tree : trees) {
            for (size_t i = 0; i < X.size(); ++i) {
                predictions[i] += learning_rate * tree->predict(X[i]);
            }
        }
        return predictions;
    }
private:
    int n_estimators;
    int max_depth;
    double learning_rate;
    vector<unique_ptr<DecisionTree>> trees;
    vector<double> train_loss;
    double calculate_loss(const vector<double>& y, const vector<double>& residuals) {
        double loss = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            loss += pow(y[i] - residuals[i], 2);
        }
        return loss / y.size();
    }
    void save_train_loss_to_file(const string& filename) {
        ofstream file(filename);
        if (file.is_open()) {
            for (double loss : train_loss) {
                file << loss << "\n";
            }
            file.close();
        } else {
            cout << "Unable to open file " << filename << endl;
        }
    }
    void save_visualization_to_file(const string& filename) {
        ofstream file(filename);
        if (file.is_open()) {
            file << "set terminal png\n";
            file << "set output 'train_loss.png'\n";
            file << "set title 'Training Loss'\n";
            file << "set xlabel 'Iteration'\n";
            file << "set ylabel 'Loss'\n";
            file << "plot 'train_loss.txt' with lines\n";
            file.close();
            system("gnuplot train_loss.plt");
        } else {
            cout << "Unable to open file " << filename << endl;
        }
    }
};

int main() {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}};
    vector<double> y = {11.0, 13.0, 15.0, 17.0, 19.0};
    XGBoost model(10, 3, 0.1);
    model.fit(X, y);
    vector<double> predictions = model.predict(X);
    for (double pred : predictions) {
        cout << pred << " ";
    }
    cout << endl;
    // Run gnuplot to visualize the training loss
    system("gnuplot -e \"plot 'train_loss.txt' with lines\"");
    return 0;
}

Corrige cette erreur


## Collaborate with GPT Engineer

This is a [gptengineer.app](https://gptengineer.app)-synced repository 🌟🤖

Changes made via gptengineer.app will be committed to this repo.

If you clone this repo and push changes, you will have them reflected in the GPT Engineer UI.

## Tech stack

This project is built with React and Chakra UI.

- Vite
- React
- Chakra UI

## Setup

```sh
git clone https://github.com/GPT-Engineer-App/xgboost-parallel-error-fix.git
cd xgboost-parallel-error-fix
npm i
```

```sh
npm run dev
```

This will run a dev server with auto reloading and an instant preview.

## Requirements

- Node.js & npm - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
