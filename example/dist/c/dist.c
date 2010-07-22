#include <stdio.h>
#include <math.h>

int main(void) {
    double x1, y1, x2, y2;
    double s;
    scanf("%lf%lf%lf%lf", &x1, &y1, &x2, &y2);
    s = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    printf("%.5f\n", s);
    return 0;
}
