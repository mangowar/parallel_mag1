#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

void axpy(double a, vector<double>& x, vector<double>& y, vector<double>& res);
double dot(vector<double>& a, vector<double>& b);
void InverseMatrix(int N, vector<double>& A, vector<int>& Ja, vector<int>& Ia, vector<double>& M_inv, vector<int>& Ia_inv, vector<int>& Ja_inv);
void SpMV(int N, vector<double>& A, vector<int>& Ja, vector<int>& Ia, vector<double>& b, vector<double>& res);
void parallel_copy(vector<double>& source,  vector<double>& dest);

bool print = false;

int Generate(vector<int>& Ia, vector<int>& Ja, int nx, int ny, int k1, int k2) {
    Ja.resize((nx+1)*(ny+1)*7);
    Ia.resize((nx+1)*(ny+1)+1);
    for(size_t i = 0; i < Ja.size(); i++) Ja[i] = -1;

    double t1 = omp_get_wtime();
    int N = (nx+1)*(ny+1);
    #pragma omp parallel for
    for(int num = 0; num < N; num++) {
            int i = num/(ny+1);
            int j = num%(ny+1);
            if(i > 0) {
                Ja[7*num+1] = (i-1)*(ny+1)+j;
            }
            if(j > 0) {
                Ja[7*num+2] = i*(ny+1)+j-1;
            }
            Ja[7*num+3] = i*(ny+1)+j;
            if(j < ny) {
                Ja[7*num+4] = i*(ny+1)+j+1;
            }
            if(i < nx) {
                Ja[7*num+5] = (i+1)*(ny+1)+j;
            }
            /* number of current element*/
            int sq_num = i*(ny+1)+j-i;
            if(j < ny && i < nx && sq_num%(k1+k2) >= k1) {
                Ja[7*num+6] = (i+1)*(ny+1)+j+1;
                Ja[7*((i+1)*(ny+1)+j+1)] = num;
            }
    }

    double t2 = omp_get_wtime();
    cout << "Generate time: " << t2 - t1 << endl;
    
    /*compress Ja*/
    size_t l = 0;
    size_t n = Ja.size();
    for(size_t r = 0; r < n; r++) {
        while(r < n && Ja[r] == -1) {
            r++;
        }
        if(r < n)
        {
            Ja[l] = Ja[r];
            l++;
        }
    }
    Ja.resize(l);
    Ia[0] = 0;
    int max = 0;
    size_t count = 1;
    for(size_t i = 0; i < l; i++) {
        if(Ja[i] < max) {
            Ia[count] = i;
            count++;
        }
        max = Ja[i];
    }
    Ia[count] = Ja.size(); 

    

    if(print){
        cout << "Nodes: " << (nx+1)*(ny+1) << endl;
        cout << "Edges: " << (Ia.back()-(nx+1)*(ny+1))/2 << endl;
    }
    return N;
}

void Fill(vector<double>& A, vector<double>& b, vector<int>& Ia, vector<int>& Ja, int N) {
    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        double diag = 0;
        for(int col = Ia[i]; col < Ia[i+1]; col++) {
            if(i != Ja[col]) {
                A[col] = cos(i*Ja[col]+i+Ja[col]);
                diag += (A[col] >= 0) ? (A[col]) : -(A[col]);
            }
        }
        b[i] = sin(i);
        diag *= 1.5;
        for(int col = Ia[i]; col < Ia[i+1]; col++) {
            if(i == Ja[col])
                A[col] = diag;
        }
        
    }
    cout << "Fill time: " << omp_get_wtime()-t1 << endl;
}

double Solve(int N, vector<double>& A,  vector<int>& Ja,  vector<int>& Ia,  vector<double>& b, vector<double>& res, double eps, int maxit) {
    vector<double> x(N, 0), x_prev(N);
    /*r = b*/
    vector<double> r(b.size()), r_prev(b.size());
    parallel_copy(b, r);
    vector<double> z(b.size()), M_inv(N), p(b.size()), p_prev(b.size()), q(b.size());
    vector<int> Ia_inv(N+1), Ja_inv(N);
    vector<double> ax(b.size()), residual(b.size());
    double t1 = omp_get_wtime(), t2;
    InverseMatrix(N, A, Ja, Ia, M_inv, Ia_inv, Ja_inv);
    int k = 0;
    double ro = 0, ro_prev, beta;
    do {
        k++;
        if(print)
            cout << "Step " << k << endl;
        SpMV(N, M_inv, Ja_inv, Ia_inv, r, z);
        ro_prev = ro;
        ro = dot(r, z);
        
        if(k == 1) {
            // p = z;
            parallel_copy(z, p);
        }
        else {
            beta = ro/ro_prev;
            /* 𝒑𝑘 = 𝒛𝑘 + 𝛽𝑘 𝒑𝑘−1*/
            // p_prev = p;
            parallel_copy(p, p_prev);
            axpy(beta, p_prev, z, p);
        }
        SpMV(N, A, Ja, Ia, p, q);
        double alpha = ro/dot(p, q);
        // x_prev = x;
        parallel_copy(x, x_prev);
        /*𝒙𝑘 = 𝒙𝑘−1 + 𝛼𝑘 𝒑𝑘*/
        axpy(alpha, p, x_prev, x);
        // r_prev = r;
        parallel_copy(r, r_prev);
        /*𝒓𝑘 = 𝒓𝑘−1 − 𝛼𝑘 𝒒𝑘*/
        axpy(-alpha, q, r_prev, r);
        SpMV(N, A, Ja, Ia, x, ax);
        axpy(-1, b, ax, residual);
        double mesure = sqrt(dot(residual, residual));
        if(print)
        {
            cout << "L2 residual " << mesure << endl;
            cout << endl;
        }
    }
    while(ro > eps*eps && k < maxit);
    
    SpMV(N, A, Ja, Ia, x, ax);
    axpy(-1, b, ax, residual);
    res = move(x);
    t2 = omp_get_wtime();
    cout << "Solve time: " << t2-t1 << endl; 
    return sqrt(dot(residual, residual));
}

void InverseMatrix(int N, vector<double>& A,  vector<int>& Ja,  vector<int>& Ia, vector<double>& M_inv, vector<int>& Ia_inv, vector<int>& Ja_inv) {
    double t1 = omp_get_wtime();
    #pragma omp parallel for default(shared)
    for(int i = 0; i < N; i++) {
        for(int col = Ia[i]; col < Ia[i+1]; col++) {
            if(i == Ja[col]) {
                M_inv[i] = 1/A[col];
                Ia_inv[i] = i;
                Ja_inv[i] = i;
            }
        }
    }
    if(print)
        cout << "InverseMatrix time: " << omp_get_wtime()-t1 << endl; 
    Ia_inv.back() = Ja_inv.size();
}

void SpMV(int N,  vector<double>& A,  vector<int>& Ja,  vector<int>& Ia, vector<double>& b, vector<double>& res) {
    // res.resize(b.size());
    for(double& i: res)
        i = 0;
    double t1 = omp_get_wtime();
    #pragma omp parallel for default(shared)
    for(int i = 0; i < N; i++) {
        double sum = 0;
        for(int col = Ia[i]; col < Ia[i+1] && col; col++) {
            sum += A[col]*b[Ja[col]];
        }
        res[i] = sum;
    }
    if(print)
        cout << "SpMV time: " << omp_get_wtime()-t1 << endl; 
}

double dot(vector<double>& a, vector<double>& b) {
    double res = 0.0;
    int n = a.size();
    double t1 = omp_get_wtime();
    #pragma omp parallel for reduction(+:res)
    for(int i = 0; i < n; i++) {
        res += a[i]*b[i];
    }
    if(print)
        cout << "dot time: " << omp_get_wtime()-t1 << endl; 
    return res;
}

void axpy(double a,  vector<double>& x,  vector<double>& y, vector<double>& res) {
    // res.resize(x.size());
    int n = x.size();
    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        res[i] = a*x[i]+y[i];
    }
    if(print)
        cout << "axpy time: " << omp_get_wtime()-t1 << endl; 
}

void parallel_copy(vector<double>& source,  vector<double>& dest) {
    int n = source.size();
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        dest[i] = source[i];
    }
}

double vec_dot(vector<double>& a, vector<double>& b) {
    double res = 0.0;
    int n = a.size();
    #pragma ivdep
    for(int i = 0; i < n; i++) {
        res += a[i]*b[i];
    }
    return res;
}

void vec_axpy(double a, vector<double>& x, vector<double>& y, vector<double>& res) {
    // res.resize(x.size());
    int n = x.size();
    #pragma unroll 10
    for(int i = 0; i < n; i++) {
        res[i] = a*x[i]+y[i];
    }
}

void Help() {
    cout << "Usage: ./a.out nx ny k1 k2 num_threads [print]" << endl;
    cout << "nx\t\tNumber of rows" << endl;
    cout << "ny\t\tNumber of collumns" << endl;
    cout << "k1\t\tNumber of square elements" << endl;
    cout << "k2\t\tNumber of pairs of triangular elements" << endl;
    cout << "num_threads\tNumber of threads which will be used" << endl;
    cout << "print\t\tPrint resulting matrixes. Optional argument" << endl;
}

int main(int argc, char* argv[]) {
    if(argc == 1) {
        Help();
        return 0;
    }
    if(argc > 1 && argc < 6) {
        cerr << "Too few arguments" << endl;
        Help();
        return 0;
    }
    if(argc == 7 && string(argv[6]) == "print") {
        print = true;
    }
    int nx, ny, k1, k2, threads;
    nx = stoi(argv[1]);
    ny = stoi(argv[2]);
    k1 = stoi(argv[3]);
    k2 = stoi(argv[4]);
    threads = stoi(argv[5]);
    if(nx < 1 || ny < 1 || (k1 <= 0 && k2 <= 0)) {
        cerr << "Wrong arguments" << endl;
        return 0;
    }

    vector<int> Ia, Ja;
    omp_set_num_threads(threads);
    int N = Generate(Ia, Ja, nx, ny, k1, k2);
    vector<double> A(Ia[N]), b(N), res;
    Fill(A, b, Ia, Ja, N);
    Solve(N, A, Ja, Ia, b, res, 0.01, 1000);

    // InverseMatrix(N, A, Ja, Ia, M, Ia_inv, Ja_Inv);
    // SpMV(N, A, Ja, Ia, b, cur_res);


    // if(print) {
        // size_t n = Ia.size();
        // size_t k = Ja.size();
        // cout << "  Ia:" << endl;
        // for(int i = 0; i < n; i++) {
        //     cout << Ia[i] << ' ';
        // }
        // cout << endl;
        // cout << "  Ja:" << endl;
        // for(int i = 0; i < k; i++) {
        //     cout << Ja[i] << ' ';
        // }
        // cout << endl;
    // }
    return 0;
}