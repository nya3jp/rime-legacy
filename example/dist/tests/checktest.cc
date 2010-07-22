#include "validator.h"

#define M 100
#define EPS 1e-12

void validate() {
    double x1 = read_double();
    read_space();
    double y1 = read_double();
    read_return();

    double x2 = read_double();
    read_space();
    double y2 = read_double();
    read_return();

    assert(0-EPS <= x1 && x1 <= M+EPS);
    assert(0-EPS <= y1 && y1 <= M+EPS);
    assert(0-EPS <= x2 && x2 <= M+EPS);
    assert(0-EPS <= y2 && y2 <= M+EPS);
}
