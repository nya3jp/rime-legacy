#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

int main() {
    double x1, y1, x2, y2;
    cin >> x1 >> y1 >> x2 >> y2;
    double s = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    cout << fixed << setprecision(5) << s << endl;
    return 0;
}
