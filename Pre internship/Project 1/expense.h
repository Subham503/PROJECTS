#ifndef EXPENSE_H
#define EXPENSE_H

#include <string>
#include <vector>
using namespace std;

struct Expense {
    string category;
    double amount;
};

void addExpense(vector<Expense>& expenses);
void viewExpenses(const vector<Expense>& expenses);

#endif
