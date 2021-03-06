#include <ml-libsvm.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <iostream>


double noise(double min, double max) {
  return min+(max-min)*(rand() / (1.0 + RAND_MAX));
}

double oracle(double x, double y, double noise_level) {
  double r = x*x+y*y;
  return exp(-2.5*r)*cos(8*sqrt(r)) + noise(-noise_level,noise_level);
}

typedef std::pair<double,double> XY;
typedef double                   Z;
typedef std::pair<XY,Z>          Data;
typedef std::vector<Data>        Basis;

// Let us redefine the oracle with our types
#define NOISE_LEVEL .2 
// Uniform(a,b) has a (a-b)^2/12 variance (here, a=-2 and b=.2).
#define NOISE_VARIANCE (4*(NOISE_LEVEL)*(NOISE_LEVEL)/12.0)  

Z Oracle(const XY& xy) {
  return oracle(xy.first,xy.second,NOISE_LEVEL);
}

// This is the same, without noise
Z CleanOracle(const XY& xy) {
  return oracle(xy.first,xy.second,0);
}


// We need a function that builds an array of svm_nodes for
// representing some input. When the input is a collection of values
// that can provide iterators, libsvm::input_of can help. Here, we
// have to write it by ourselves.


int nb_nodes_of(const XY& xy) {
  return 3;
}

void fill_nodes(const XY& xy,struct svm_node* nodes) {
  nodes[0].index = 1;
  nodes[0].value = xy.first;  // x 
  nodes[1].index = 2;
  nodes[1].value = xy.second; // y
  nodes[2].index = -1;        // end
}

const XY& input_of(const Data& data) {return data.first;}
double output_of(const Data& data) {return data.second;}

// This is gnuplots a function
#define PLOT_STEP .04
template<typename Func>
void gnuplot(std::string filename,
	     std::string title,
	     const Func& f) {
  XY x;
  std::ofstream file;

  file.open(filename.c_str());
  if(!file) {
    std::cerr << "Cannot open \"" << filename << "\"." << std::endl;
    return;
  }

  file << "set hidden3d" << std::endl
       << "set title \"" << title << "\"" << std::endl
       << "set view 41,45" << std::endl
       << "set xlabel \"x\"" << std::endl
       << "set ylabel \"y\"" << std::endl
       << "set zlabel \"z\"" << std::endl
       << "set ticslevel 0" << std::endl
       << "splot '-' using 1:2:3 with lines notitle" << std::endl;

  for(x.first=-1;x.first<=1;x.first+=PLOT_STEP,file << std::endl)
    for(x.second=-1;x.second<=1;x.second+=PLOT_STEP,file << std::endl)
      file << x.first << ' ' << x.second << ' ' << f(x);

  file.close();
  std::cout << "Gnuplot file \"" << filename << "\" generated." << std::endl;
}

int main(int argc, char* argv[]) {

  // Let us make libsvm quiet
  ml::libsvm::quiet();

  if(argc != 2) {
    std::cerr << "Usage : " << argv[0] << " <nb-samples>" << std::endl;
    return 1;
  }

  int nb_samples = atoi(argv[1]);
  if(nb_samples < 50) {
    std::cout << "I am using at least 50 samples." << std::endl;
    nb_samples = 50;
  }
  
  try {

    // Let us collect samples.
    
    Basis basis;
    Basis::iterator iter,end;

    basis.resize(nb_samples);
    for(iter = basis.begin(), end = basis.end(); iter != end; ++iter) {
      XY xy(noise(-1,1),noise(-1,1));
      *iter = Data(xy,Oracle(xy));
    }
    // Let us set configure a svm

    struct svm_parameter params;
    ml::libsvm::init(params);
    params.svm_type    = EPSILON_SVR;
    params.eps         = 1e-5;         // numerical tolerence
    // TODO: Tune the parameters
    params.kernel_type = RBF;          // RBF kernel
    params.p           = .05;          // epsilon
    params.gamma       = 10;           // k(u,v) = exp(-gamma*(u-v)^2)
    params.C           = 1;

    // This sets up a svm learning algorithm.
    auto learner = ml::libsvm::learner(params,nb_nodes_of,fill_nodes);

    // Let us train it and get some predictor f. f is a function, as the oracle.
    std::cout << "Learning..." << std::endl;
    auto f = learner(basis.begin(),basis.end(),input_of,output_of);

    // Let us plot the result.
    gnuplot("clean-oracle.plot","Clean oracle",CleanOracle);
    gnuplot("oracle.plot","Oracle",Oracle);
    gnuplot("prediction.plot","SVM prediction",f);

    // All libsvm functions related to svm models are implemented.
    std::cout << std::endl
    	      << "There are " << f.get_nr_sv() << " support vectors." << std::endl;

    // We can compute the empirical risk with ml tools.
    double risk = ml::risk::empirical(f,
    				      basis.begin(),basis.end(),
    				      input_of, output_of,
    				      ml::loss::Quadratic<double>());
    std::cout << "Empirical quadratic risk : " << risk << std::endl
    	      << "               i.e error : " << sqrt(risk) << std::endl;
    
    // Let us use a cross-validation procedure
    risk = ml::risk::cross_validation(learner,
    				      ml::partition::kfold(basis.begin(),basis.end(),10),
    				      input_of,output_of,
    				      ml::loss::Quadratic<double>(),
    				      true);
    std::cout << "Real quadratic risk estimation : " << risk << std::endl
    	      << "                     i.e error : " << sqrt(risk) << std::endl
    	      << "      noise standard deviation : " << sqrt(NOISE_VARIANCE) << std::endl;


    // Now, let us save our predictor.
    f.save_model("f.pred");

    // We can load it and use it on another databasis. The newly
    // created predictor (from scratch) predictor must be provided
    // with the required conversion functions.
    ml::libsvm::Predictor<XY> g(nb_nodes_of,fill_nodes);
    g.load_model("f.pred");

    // We re-sample the data basis.
    for(iter = basis.begin(), end = basis.end(); iter != end; ++iter) {
      XY xy(noise(-1,1),noise(-1,1));
      *iter = Data(xy,Oracle(xy));
    }

    // We compute the risk of the loaded predictor on this databasis.
    risk = ml::risk::empirical(g,
    			       basis.begin(),basis.end(),
    			       input_of, output_of,
    			       ml::loss::Quadratic<double>());
    std::cout << "Empirical risk (loaded predictor) : " << risk << std::endl
    	      << "                        i.e error : " << sqrt(risk) << std::endl;

  }
  catch(ml::exception::Any& e) {
    std::cout << e.what() << std::endl;
  }
  
  return 0;
}
