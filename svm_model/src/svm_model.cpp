#include <ml-libsvm.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <iostream>

typedef std::pair<double,double> XY;
typedef double                   Z;
typedef std::pair<XY,Z>          Data;
typedef std::vector<Data>        Basis;

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
#define PLOT_STEP .1
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

  for(x.first=-5;x.first<=5;x.first+=PLOT_STEP,file << std::endl)
    for(x.second=-5;x.second<=5;x.second+=PLOT_STEP,file << std::endl)
      file << x.first << ' ' << x.second << ' ' << f(x);

  file.close();
  std::cout << "Gnuplot file \"" << filename << "\" generated." << std::endl;
}


int number_samples(const char* filename){
	std::ifstream file;
	file.open(filename,std::ios::in);
	if(!file) {
		std::cerr << "Cannot open \"" << filename << "\"." << std::endl;
		return 1;
	}
	int lines_count =0;
	std::string line;
	while (std::getline(file , line))
		++lines_count;
	return lines_count;
}


int main(int argc, char* argv[]) {

  // Let us make libsvm quiet
  ml::libsvm::quiet();

  if(argc != 2) {
    std::cerr << "Usage : " << argv[0] << " for data-samples file" << std::endl;
    return 1;
  }
  
  int nb_samples = number_samples(argv[1]);
  std::cout << "There are :" << nb_samples << " samples" << std::endl;
  
   try {

    // Let us collect samples.
    
    Basis basis;
    Basis::iterator iter,end;

    basis.resize(nb_samples);
    std::ifstream dataset;
    dataset.open(argv[1],std::ios::in);
	if(!dataset) {
		std::cerr << "Cannot open \"" << argv[1] << "\"." << std::endl;
		return 1;
	}
	
	// special case for dataset_100 being too big : we take one point over 50
	if (argv[1]== std::string("dataset_100")){
		std::ofstream smalldataset;
		smalldataset.open("smalldataset_100");
		int count = 0;
		double x,y,z;
		while (dataset >> z >> x >> y){
			if (count%50 == 0 ){
				smalldataset << z << " "<< x << " " << y << std::endl;
			}
			++ count;
		}
		smalldataset.close();
		dataset.close();
		dataset.open("smalldataset_100",std::ios::in);
		nb_samples = number_samples("smalldataset_100");
		basis.resize(nb_samples);
		std::cout << "After reducing dataset there are :" << nb_samples << " samples" << std::endl;
	}
	
    for(iter = basis.begin(), end = basis.end(); iter != end; ++iter) {
      double x,y,z;
	  dataset >> z >> x >> y;
      XY xy(x,y);
      *iter = Data(xy,z);
    }
    dataset.close();
    // Let us set configure a svm

    struct svm_parameter params;
    ml::libsvm::init(params);
    params.svm_type    = EPSILON_SVR;
    params.eps         = 1e-5;         // numerical tolerence
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
    gnuplot("mapmodel.plot","SVM model of the 3D map",f);

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
    	      << "                     i.e error : " << sqrt(risk) << std::endl;


    // Now, let us save our predictor.
    f.save_model("f.pred");

    // We can load it and use it on another databasis. The newly
    // created predictor (from scratch) predictor must be provided
    // with the required conversion functions.
    ml::libsvm::Predictor<XY> g(nb_nodes_of,fill_nodes);
    g.load_model("f.pred");
	
	// second dataset to cross validate
    // We re-sample the data basis.
    dataset.open("/home/GTL/agout/catkin_ws/dataset_bayes2",std::ios::in);
    if(!dataset) {
		std::cerr << "Cannot open \"" << argv[1] << "\"." << std::endl;
		return 1;
	}
    for(iter = basis.begin(), end = basis.end(); iter != end; ++iter) {
	  double x,y,z;
	  dataset >> z >> x >> y;
      XY xy(x,y);
      *iter = Data(xy,z);
    }
    dataset.close();

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
