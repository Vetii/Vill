#include <algorithm>
#include <gecode/int.hh> // Integer variables
#include <gecode/float.hh> // Float variables
#include <gecode/search.hh> // Search support
#include <gecode/minimodel.hh> // Minimodel
#include "linalg.h"
#include "neuralnet.h"

NeuralNet::NeuralNet(Relation<FloatVector, FloatVector> relation,
        std::vector<unsigned int> structure) :
    _stages(),
    _tMatrices(),
    _input(),
    _output() {

    // Check the structure
    std::vector<unsigned int> strkt = checkStructure(relation, structure);

    // Get the range of the data
    std::pair<float, float> fullBounds= getFullBounds(relation);

    // Generate stage for each layer
    std::vector<Gecode::Matrix<Gecode::FloatVarArray>> stageMatrices;
    for(unsigned int nbNeurons : strkt) {
       stageMatrices.push_back(genStage(
                   fullBounds.first,
                   fullBounds.second,
                   relation.size(),
                   nbNeurons
                   )); 
    }

    // Setup the pre-processed matrices
    std::vector<Gecode::Matrix<Gecode::FloatVarArray>> PPrMatrices;
    for(unsigned int i = 0; i < _stages.size(); ++i) {
        //Gecode::FloatVarArray ppr(*this, _stages[i]);
        Gecode::FloatVarArray ppr(*this,
                _stages[i].size(),
                fullBounds.first,
                fullBounds.second);
        _preprocessed.push_back(ppr); 

        // All stages are equal to a function of their
        // pre-processing representation
        // EXCEPT:
        // The first stage has no pre-processed value
        // Thas stage is a linear function of its preprocessed value
        if(i == 0 || i == _stages.size() - 1) {
            LinAlg::map(_stages[i], _preprocessed[i],
                    [this](Gecode::FloatVar x, Gecode::FloatVar y) {
                    rel(*this, x == y);
                    });
        } else {
            LinAlg::map(_stages[i], _preprocessed[i],
                    [this](Gecode::FloatVar x, Gecode::FloatVar y) {
                    rel(*this, x == atan(y));
                    });
        }

        Gecode::Matrix<Gecode::FloatVarArray> mat(ppr,
                stageMatrices[i].width(),
                stageMatrices[i].height());

        PPrMatrices.push_back(mat);
    }


    // Set up transformation matrices
    std::vector<Gecode::Matrix<Gecode::FloatVarArray>> tMatrices;
    for(unsigned int i = 1; i < strkt.size(); ++i) {
        Gecode::Matrix<Gecode::FloatVarArray> tMat = 
            genTransition(
                    fullBounds.first, 
                    fullBounds.second,
                    PPrMatrices.at(i - 1), 
                    PPrMatrices.at(i));

        // curr_stage = previous_stage * mat
        LinAlg::product(*this, PPrMatrices.at(i), stageMatrices.at(i - 1), tMat);
        //LinAlg::diagonal(*this, tMat);

        tMatrices.push_back(tMat);
    }

    // The first stage matches the domain of relation
    for(unsigned int row = 0; row < relation.domain().size(); ++row) {
        FloatVector rowVals = relation.domain().at(row);
        for(unsigned int col = 0; col < rowVals.size(); ++col) {
            rel(*this, stageMatrices.at(0)(col, row) == rowVals[col]);
        }
    }
    
    // The first stage matches the domain of relation
    for(unsigned int row = 0; row < relation.range().size(); ++row) {
        FloatVector rowVals = relation.range().at(row);
        for(unsigned int col = 0; col < rowVals.size(); ++col) {
            rel(*this, stageMatrices.at(_stages.size() - 1)(col, row) == rowVals[col]);
        }
    }
    // The last stage  matches the range  of relation

    // BRANCHING.
    Gecode::FloatVarArgs branchTMat;
    for(unsigned int i = 0; i < _tMatrices.size(); ++i) {
        branchTMat << _tMatrices[i];

    }
    Gecode::branch(*this, branchTMat,
            Gecode::FLOAT_VAR_SIZE_MAX(),
            Gecode::FLOAT_VAL_SPLIT_MIN());

    Gecode::FloatVarArgs branchStages;
    Gecode::FloatVarArgs branchPPR;
    for(unsigned int i = 0; i < _stages.size(); ++i) {
        branchStages << _stages[i];
        branchPPR    << _preprocessed[i];
    }

    Gecode::branch(*this, branchStages,
            Gecode::FLOAT_VAR_SIZE_MAX(),
            Gecode::FLOAT_VAL_SPLIT_MIN());
    Gecode::branch(*this, branchPPR,
            Gecode::FLOAT_VAR_SIZE_MAX(),
            Gecode::FLOAT_VAL_SPLIT_MIN());
}

std::vector<unsigned int> NeuralNet::checkStructure(
        const Relation<FloatVector, FloatVector> &relation,
        const std::vector<unsigned int> &structure) {
    
    // The number of inputs should match the number of input neurons
    // the number of outputs should match the number of output neurons
    unsigned int inputDim = relation.at(0).first.size();
    unsigned int outputDim= relation.at(0).second.size();

    std::vector<unsigned int> strkt;

    // Check the structure: input of the network
    // has the right number of dimensions
    if(structure.at(0) != inputDim) {
        strkt.push_back(inputDim);
    }
    
    // Insert the rest of the structure
    strkt.insert(strkt.end(), structure.begin(), structure.end());

    // Check the structure: output of the network
    // has the right number of dimensions
    if(structure.at(structure.size() - 1) != outputDim) {
        strkt.push_back(outputDim);
    }

    return strkt;

}

// Get the full range of the data
std::pair<float, float> NeuralNet::getFullBounds(
        const Relation<FloatVector, FloatVector> &relation) {

    // get the lowest and biggest values of the input range.
    std::pair<FloatVector, FloatVector> inputBounds = relation.getInputBounds();
    float inputMin = *std::min_element(inputBounds.first.begin(),
            inputBounds.first.end());
    float inputMax = *std::max_element(inputBounds.second.begin(),
            inputBounds.second.end());

    // Get the value with the biggest amplitude in input.
    float inputAbsMax = std::max(std::abs(inputMin), std::abs(inputMax));

    // get the lowest and biggest values of the output range.
    std::pair<FloatVector, FloatVector> outputBounds = relation.getOutputBounds();
    float outputMin = *std::min_element(outputBounds.first.begin(),
            outputBounds.first.end());
    float outputMax = *std::max_element(outputBounds.second.begin(),
            outputBounds.second.end());

    // Get the value with biggest amplitude in output.
    float outputAbsMax = std::max(std::abs(outputMin), std::abs(outputMax));

    // Get the one value that is REALLY BIGGER
    float bound = std::max(inputAbsMax, outputAbsMax);

    return std::make_pair(-10 * bound, 10 * bound);
}

void NeuralNet::setupStages(
        const Relation<FloatVector, FloatVector> &relation,
        const std::pair<float, float> fullBounds,
        const std::vector<unsigned int> &structure) {

    unsigned int nbRows = relation.size();
    unsigned int nbCols = 0;

    // Create the intermediate stages.
    for(unsigned int nbNeurons : structure) {
        nbCols = nbNeurons; 

        Gecode::FloatVarArray arr (*this, nbRows * nbCols, 
                fullBounds.first, fullBounds.second);

        _stages.push_back(arr);
    }
}

void NeuralNet::setupPreProcessed(
        const std::vector<Gecode::FloatVarArray> &stages) {
    
    // Copy the array of stages as _preprocessed
    // Except the first one
    _preprocessed.reserve(stages.size());
    for(unsigned int i = 0 ; i < stages.size(); ++i) {
        _preprocessed.push_back(Gecode::FloatVarArray(*this, stages[i]));
    }

}

void NeuralNet::setupTransitions(
        const Relation<FloatVector, FloatVector> &relation,
        const std::pair<float, float> fullBounds,
        const std::vector<unsigned int> &structure) {

    for (unsigned int i = 1; i < structure.size(); ++i) {
        unsigned int nbRows = structure.at(i - 1);
        unsigned int nbCols = structure.at(i);

        Gecode::FloatVarArray arr(*this, nbRows * nbCols,
                fullBounds.first, fullBounds.second);

        _tMatrices.push_back(arr);
    }

}

// Given a vector of arrays and a vector of dimensions, build
// matrices of correct sizes
std::vector<Gecode::Matrix<Gecode::FloatVarArray>> 
    NeuralNet::getMatrices(
        const std::vector<Gecode::FloatVarArray> &data,
        const Relation<unsigned int, unsigned int> &heightWidths) {
    // Check the two vectors have the same size
    assert(data.size() == heightWidths.size());

    std::vector<Gecode::Matrix<Gecode::FloatVarArray>> matrices;

    for (unsigned int i = 0; i < data.size(); ++i) {
        Gecode::Matrix<Gecode::FloatVarArray> mat(data.at(i), 
                heightWidths.at(i).second,
                heightWidths.at(i).first);
        matrices.push_back(mat);
    }

    return matrices;
}

Gecode::Matrix<Gecode::FloatVarArray> NeuralNet::genStage(
        float minVal, float maxVal,
        unsigned int relationSize,
        unsigned int nbNeurons) {

    Gecode::FloatVarArray arr(*this, relationSize * nbNeurons,
            minVal, maxVal);

    _stages.push_back(arr);
    
    Gecode::Matrix<Gecode::FloatVarArray> mat(arr,
            nbNeurons, relationSize);

    return mat;
}

Gecode::Matrix<Gecode::FloatVarArray> NeuralNet::genTransition(
        float minVal, float maxVal,
        const Gecode::Matrix<Gecode::FloatVarArray> &A,
        const Gecode::Matrix<Gecode::FloatVarArray> &B) {

    Gecode::FloatVarArray arr(*this, A.width() * B.width(),
            minVal, maxVal);

    // Useless link with the home 
    _tMatrices.push_back(arr);

    const Gecode::Matrix<Gecode::FloatVarArray> mat(arr,
            B.width(), A.width());

    return mat;

}

void NeuralNet::print(void) const {
    NeuralNet::print(std::cout);
}

void NeuralNet::print(std::ostream& os) const {
    for(unsigned int i = 0; i < _stages.size(); ++i) {
        os << "PREPROCESSED: " << i << std::endl;
        os << _preprocessed[i] << std::endl;
        os << "STAGE: " << i << std::endl;
        os << _stages[i]     << std::endl;
        os << std::endl;
    }
    os << std::endl;
    for(unsigned int i = 0; i < _tMatrices.size(); ++i) {
        os << "TMATRIX: " << i << std::endl;
        os << _tMatrices[i]    << std::endl;
    }
    os << std::endl;
}
