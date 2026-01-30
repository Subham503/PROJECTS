#include "expense.h"
#include <iostream>
#include <vector>
using namespace std;

void addExpense(vector<Expense>& expenses) {
    Expense e;
    cout << "Enter category: ";
    cin >> e.category;
    cout << "Enter amount: ";
    cin >> e.amount;

    expenses.push_back(e);
    cout << "Expense added successfully!\n";
}

void viewExpenses(const vector<Expense>& expenses) {
    if (expenses.empty()) {
        cout << "No expenses yet!\n";
        return;
}

cout << "Expenses List:\n";
    for (size_t i = 0; i < expenses.size(); i++) {
        cout << i + 1 << ". " << expenses[i].category 
             << " - " << expenses[i].amount << endl;
    }
}