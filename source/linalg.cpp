#include "linalg.h"

// post the constraint that C = AB 
void LinAlg::product (
        Gecode::Home home,
        const Gecode::Matrix<Gecode::FloatVarArray> &C,
        const Gecode::Matrix<Gecode::FloatVarArray> &A, 
        const Gecode::Matrix<Gecode::FloatVarArray> &B) {

    // A : m x n
    // B : n x p
    // C = AB : m x p

    int m = A.height();
    int n = A.width();
    int p = B.width();

    assert(A.width() == B.height());
    assert(C.width() == p && C.height() == m);

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < p; ++j) {
            product(home, C(j, i), A.row(i), B.col(j));
        }
    }
}

void LinAlg::product(Gecode::Home home,
        const Gecode::FloatVar &C,
        const Gecode::FloatVarArgs &A,
        const Gecode::FloatVarArgs &B) {

    assert(A.size() == B.size());
    
    // The termwise product of vectors
    Gecode::FloatVarArgs prod(home, A.size(),
            Gecode::Float::Limits::min,
            Gecode::Float::Limits::max);


    for(int i = 0; i < A.size(); ++i) {
        rel(home, prod[i] == A[i] * B[i]); 
    }

    linear(home, prod, Gecode::FRT_EQ, C);

}

void LinAlg::diagonal(Gecode::Home home, 
        const Gecode::Matrix<Gecode::FloatVarArray> &A) {

    for(int i = 0; i < A.height(); ++i) {
        for(int j = 0; j < A.width(); ++j) {
            if(i != j) {
                rel(home, A(j, i) == 0);
            } 
        }
    }

}

void LinAlg::addition (
        Gecode::Home home,
        const Gecode::Matrix<Gecode::FloatVarArray> &C,
        const Gecode::Matrix<Gecode::FloatVarArray> &A, 
        const Gecode::Matrix<Gecode::FloatVarArray> &B) {

    // A : m x n
    // B : m x n
    // C = A + B : m x n

    int m = A.height();
    int n = A.width();

    assert(A.width()  == B.height() && A.width()  == C.width());
    assert(A.height() == B.height() && A.height() == C.height());

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            rel(home, A(j, i) + B(j, i) == C(j, i));
        }
    }
}

void LinAlg::map (const Gecode::FloatVarArgs &v1,
                  const Gecode::FloatVarArgs &v2,
                  std::function<void(Gecode::FloatVar, Gecode::FloatVar)> f) {

    assert(v1.size() == v2.size());

    for (int i = 0; i < v1.size(); ++i) {
        // Apply constraint between the two variables
        f(v1[i], v2[i]);
    }

}
