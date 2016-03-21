#include <memory>
#include <algorithm>
#include <vector>
#include <utility>
#include <gecode/int.hh> // Integer variables
#include <gecode/float.hh> // Float variables
#include <gecode/search.hh> // Search support
#include <gecode/minimodel.hh> // Minimodel

#pragma once

template <class T> 
class ComparableVector : public std::vector<T> {
    bool operator <(const ComparableVector &v) {
        unsigned int sz = std::min(this->size(), v.size());
        for(unsigned int i = 0; i < sz; ++i) {
            if(this->at(i) < v.at(i)) {
                return true;
            }
            if(this->at(i) > v.at(i)) {
                return false;
            }
        }
        // all tested values seem equal 
        if(this->size() < v.size()) { return true; }
        return false;
    }
};

typedef ComparableVector<float> FloatVector;

// A class representing a relation,
// Actually, a total order
template <class T1, class T2>
class Relation { 
    private:
        //std::vector<std::pair<T1, T2>> _data;
        std::vector<T1> _domain;
        std::vector<T2> _range;

    public:
        Relation(std::vector<std::pair<T1, T2>> data) :
            _domain(),
            _range() {
                for(std::pair<T1, T2> p : data) {
                    _domain.push_back(p.first);
                    _range.push_back(p.second);
                }
            }

        std::vector<T1> domain() const {
            return _domain;
        }

        std::vector<T2> range() const { 
            return _range;
        }

        std::pair<T1, T1> getInputBounds() const {
            T1 min = _domain.at(0);
            T1 max = _domain.at(0);
            for(T1 val : _domain) {
                min = std::min(val, min); 
                max = std::max(val, max);
            }
            return std::make_pair(min, max);
        }

        std::pair<T2, T2> getOutputBounds() const {
            T2 min = _range.at(0);
            T2 max = _range.at(0);
            for(T2 val : _range) {
                min = std::min(val, min); 
                max = std::max(val, max);
            }
            return std::make_pair(min, max);
        }

        std::pair<T1, T2> at(unsigned int n) {
            return std::make_pair(_domain.at(n), _range.at(n));
        }

        const std::pair<T1, T2> at(unsigned int n) const {
            return std::make_pair(_domain.at(n), _range.at(n));
        }

        unsigned int size() const { return _domain.size(); }

};

class NeuralNet : public Gecode::Space {
    private:
        // Set of pairs of vectors as the relation
        // std::shared_ptr<Relation<Vector, Vector>> > _relation;
        // List of integers for the structure (each integer being the size of
        // the layers
        // std::shared_ptr<std::vector<unsigned int>> _structure;
        
    protected:
        // Storage of data at all processing stages 
        // (A vector of matrices)
        std::vector<Gecode::FloatVarArray> _stages;

        // Storage for all the pre-threshold values
        // copies of _stages
        std::vector<Gecode::FloatVarArray> _preprocessed;

        // Storage of all transformation matrices
        std::vector<Gecode::FloatVarArray> _tMatrices;

        // Input data matrix
        Gecode::FloatVarArgs _input;

        // Output data matrix
        Gecode::FloatVarArgs _output;

    public:
        // CONSTRUCTORS 
        NeuralNet(Relation<FloatVector, FloatVector> relation,
                std::vector<unsigned int> structure);

        NeuralNet(bool share, NeuralNet& n) : Gecode::Space(share, n),
            _stages(n._stages), 
            _preprocessed(n._preprocessed),
            _tMatrices(n._tMatrices),
            _input(n._input),
            _output(n._output) {
            // Update stages separately
            for(unsigned int i = 0; i < n._stages.size(); ++i) {
                _stages.at(i).update(*this, share, n._stages.at(i));
            }
            // Update pre-processed stages
            for(unsigned int i = 0; i < n._preprocessed.size(); ++i) {
                _preprocessed.at(i).update(*this, share, n._preprocessed.at(i));
            }
            // Update the transformation matrices.
            for(unsigned int i = 0; i < n._tMatrices.size(); ++i) {
                _tMatrices.at(i).update(*this, share, n._tMatrices.at(i));
            }
        }

        // PRINTING
        void print(void) const;

        void print(std::ostream& os) const;
        
        // COPY
        Gecode::Space* copy (bool share) {
            return new NeuralNet(share, *this);
        }

        std::vector<unsigned int> checkStructure(
                const Relation<FloatVector, FloatVector> &relation,
                const std::vector<unsigned int> &structure);

        std::pair<float, float> getFullBounds(
                const Relation<FloatVector, FloatVector> &relation);

        void setupStages(
                const Relation<FloatVector, FloatVector> &relation,
                const std::pair<float, float> fullBounds,
                const std::vector<unsigned int> &structure);

        void setupPreProcessed(
                const std::vector<Gecode::FloatVarArray> &stages);

        void setupTransitions(
                const Relation<FloatVector, FloatVector> &relation,
                const std::pair<float, float> fullBounds,
                const std::vector<unsigned int> &structure);
        
        // Transforms arrays into matrices
        std::vector<Gecode::Matrix<Gecode::FloatVarArray>> 
            getMatrices(
                    const std::vector<Gecode::FloatVarArray> &data,
                    const Relation<unsigned int, unsigned int> &heightWidths);

        Gecode::Matrix<Gecode::FloatVarArray> genStage(
                float minVal, float maxVal,
                unsigned int relationSize,
                unsigned int nbNeurons);


        // Creates a proper transition matrix "between" two stages.
        Gecode::Matrix<Gecode::FloatVarArray> genTransition(
                float minVal, float maxVal,
                const Gecode::Matrix<Gecode::FloatVarArray> &A,
                const Gecode::Matrix<Gecode::FloatVarArray> &B);

        const Gecode::FloatVarArray & getInputArr() {
            return _stages.at(0);
        }

        const Gecode::FloatVarArray & getOutputArr() { 
            return _stages.at(_stages.size() - 1);
        }


};

