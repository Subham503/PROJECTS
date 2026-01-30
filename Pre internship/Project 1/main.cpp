#include <iostream>
#include <vector>
#include "expense.h"
using namespace std;

int main() {
    vector<Expense> expenses;
    int choice;

    do {
        cout << "\n1. Add Expense\n";
        cout << "2. View Expenses\n";
        cout << "3. Exit\n";
        cout << "Enter choice: ";
        cin >> choice;

        if (choice == 1) {
            addExpense(expenses);
        } else if (choice == 2) {
            viewExpenses(expenses);
        } else if (choice == 3) {
            cout << "Exiting...\n";
        } else {
            cout << "Invalid choice, try again.\n";
        }
    } while (choice != 3);

    return 0;
}
