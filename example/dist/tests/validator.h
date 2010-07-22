#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <iterator>
#include <functional>
#include <utility>
#include <numeric>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <cstring>
#include <climits>
using namespace std;

#include <unistd.h>

#define REP(i,n) for(int i = 0; i < (int)(n); i++)
#define FOR(i,c) for(__typeof((c).begin()) i = (c).begin(); i != (c).end(); ++i)
#define ALLOF(c) ((c).begin()), ((c).end())

#undef assert
#define assert(x) assert_impl(x, __FILE__, __LINE__)
#define cassert(x) assert_impl(x, file, line)
#define assert_impl(x,file,line) do { if (!(x)) { printf("CHECK FAILED at %s:%d", file, line); exit(1); } } while(0)

#define assure_token() assure_token_impl(__FILE__, __LINE__)
void assure_token_impl(const char* file, const int line) {
    char c;
    int r = scanf("%c", &c);
    cassert(r == 1);
    cassert(!isspace(c));
    ungetc(c, stdin);
}

#define read_space() read_space_impl(__FILE__, __LINE__)
void read_space_impl(const char* file, const int line) {
    char c;
    int r = scanf("%c", &c);
    cassert(r == 1);
    cassert(c == ' ');
}

#define read_return() read_return_impl(__FILE__, __LINE__)
void read_return_impl(const char* file, const int line) {
    char c;
    int r = scanf("%c", &c);
    cassert(r == 1);
    cassert(c == '\n');
}

#define read_int(lo, hi) read_int_impl(lo, hi, __FILE__, __LINE__)
int read_int_impl(int lo, int hi, const char* file, const int line) {
    assure_token_impl(file, line);
    string s;
    cin >> s;
    cassert(cin);
    REP(i, s.size())
        cassert((i == 0 && s[i] == '-') || ('0' <= s[i] && s[i] <= '9'));
    cassert(s.size() <= 12);
    long long int tmp;
    istringstream is(s);
    is >> tmp;
    cassert(lo <= tmp && tmp <= hi);
    return tmp;
}

#define read_double() read_double_impl(__FILE__, __LINE__)
double read_double_impl(const char* file, const int line) {
    assure_token_impl(file, line);
    string s;
    cin >> s;
    cassert(cin);
    bool dot = false;
    REP(i, s.size()) {
        cassert((i == 0 && s[i] == '-') || ('0' <= s[i] && s[i] <= '9') || (s[i] == '.'));
        if (s[i] == '.') {
            cassert(!dot);
            cassert(i >= 1 && i < (int)s.size()-1);
            if (s[0] == '-')
                cassert(i >= 2);
            dot = true;
        }
    }
    double tmp;
    istringstream is(s);
    is >> tmp;
    return tmp;
}

#define read_string(lo, hi) read_string_impl(lo, hi, __FILE__, __LINE__)
string read_string_impl(int lo, int hi, const char* file, const int line) {
    assure_token_impl(file, line);
    string s;
    cin >> s;
    cassert(lo <= (int)s.size() && (int)s.size() <= hi);
    return s;
}

#define read_char() read_char_impl(__FILE__, __LINE__)
char read_char_impl(const char* file, const int line) {
    assure_token_impl(file, line);
    int c = cin.get();
    assert(isprint(c));
    return (char)c;
}

void assure_eof() {
    char c;
    assert(scanf("%c", &c) == EOF);
}

int main() {

    void validate();
    validate();

    assure_eof();
    //puts("OK");
    return 0;
}
