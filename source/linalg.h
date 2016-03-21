#include <functional>
#include <gecode/float.hh>
#include <gecode/minimodel.hh>

#pragma once 


// A bunch of constraint for linear algebra functions in Gecode

namespace LinAlg {
    
        // Matrix multiplication
        // C = AB 
        void product (
                Gecode::Home home,
                const Gecode::Matrix<Gecode::FloatVarArray> &C,
                const Gecode::Matrix<Gecode::FloatVarArray> &A, 
                const Gecode::Matrix<Gecode::FloatVarArray> &B);

        void product(
                Gecode::Home home,
                const Gecode::FloatVar &C,
                const Gecode::FloatVarArgs &A,
                const Gecode::FloatVarArgs &B);


        // Forces a matrix to be diagonal
        void diagonal(Gecode::Home home, 
                const Gecode::Matrix<Gecode::FloatVarArray> &A);

        // C = A + B
        void addition(
                Gecode::Home home,
                const Gecode::Matrix<Gecode::FloatVarArray> &C,
                const Gecode::Matrix<Gecode::FloatVarArray> &A, 
                const Gecode::Matrix<Gecode::FloatVarArray> &B);

        // C = f(A), where f is a component-wise function
        void map(
                const Gecode::FloatVarArgs &v1,
                const Gecode::FloatVarArgs &v2,
                std::function<void(Gecode::FloatVar, Gecode::FloatVar)> f);

        template <class InputIter1, class InputIter2,
                  class Function  , class OutputIter>
        OutputIter zip(InputIter1 begin1, InputIter1 end1,
                       InputIter2 begin2, InputIter2 end2,
                       OutputIter target, Function f) {
            assert(std::distance(begin1, end1) == std::distance(begin2, end2));
            
            while(begin1 != end1) {
               *target = f(*begin1, *begin2);
               ++target; ++begin1; ++begin2;
            }

            return target;
        }

}

