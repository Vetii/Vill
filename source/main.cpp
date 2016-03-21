#include "neuralnet.h"
//#include <gecode/search.hh> // Search support
#include <gecode/gist.hh>

int main(int argc, char *argv[])
{
    std::vector<std::pair<FloatVector, FloatVector>> rel;
    
    for(float x = 1; x < 10; x += 1) {
        for(float y = 1; y < 10; y += 1) {
            FloatVector point;
            point.push_back(x);
            point.push_back(y);
            point.push_back(1);

            FloatVector res;
            res.push_back(x);
            res.push_back(y + 3);
            res.push_back(1);

            rel.push_back(std::make_pair(point, res));
        }
    }

    Relation<FloatVector, FloatVector> relation(rel);

    std::vector<unsigned int> structure = { 3, 3};

    NeuralNet* nn = new NeuralNet(relation, structure);
    nn->status();
    nn->print();
    //nn->print();
    Gecode::BAB<NeuralNet> se(nn);
    Gecode::Gist::Print<NeuralNet> p("Print solution");
    Gecode::Gist::Options o;
    o.inspect.click(&p);
    Gecode::Gist::bab(nn, o);
    delete nn;

    /*
    if(NeuralNet* nn1 = se.next()) {
        nn1->print();
        delete nn1;
    }*/

    return 0;
}
