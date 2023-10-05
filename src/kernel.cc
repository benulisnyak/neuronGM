/*
 * Copyright (c) 2013-2023 Javier G. Orlandi <javier.orlandi@ucalgary.ca>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <iostream>
#include <sys/time.h>
#include "gsl/gsl_randist.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_statistics_double.h"
#include "gsl/gsl_sf_pow_int.h"
#include "kernel.h"

// TODOS
// on percolation run check for x<0 condition

using namespace std;
using namespace Eigen;

Kernel::Kernel()
{
  rng = NULL;
  tmpDebug = false;
}

void Kernel::initialize(Sim& sim)
{
  // Load simulation parameters
  sim.get("x", x);
  sim.get("m0", m0);
  sim.get("p0", p0, 0);
  sim.get("p1", p1, 0);
  sim.get("steps", steps, 100000);
  sim.get("maxVectorSize", maxVectorSize, 5000000);
  sim.get("networkLinksFile", networkLinksFile);
  sim.get("networkWeightsFile", networkWeightsFile);
  sim.get("networkSize", networkSize);
  N = networkSize; 
  sim.get("outputBaseFile", outputBaseFile);
  sim.get("outputSpikes", outputSpikes);
  sim.get("outputMeans", outputMeans);
  sim.get("outputPercolationSteps", outputPercolationSteps);
  sim.get("outputExtension", outputExtension);
  sim.get("outputAvalancheStatistics", outputAvalancheStatistics);
  sim.get("outputPercolationPoint", outputPercolationPoint);
  
  sim.get("avalancheOnlyHack", avalancheOnlyHack, false);
  
  sim.get("networkOutputLinksFile", networkOutputLinksFile);
  sim.get("networkOutputWeightsFile", networkOutputWeightsFile);
  sim.get("calculateAvalanches", calculateAvalanches, true);
  sim.get("saveFullAvalancheData", saveFullAvalancheData, true);
  
  sim.get("calculateEffectiveNetwork", calculateEffectiveNetwork, true);
  sim.get("calculateMeans", calculateMeans, true);
  sim.get("saveSpikesData", saveSpikesData, true);
  sim.get("fastAvalancheCalculation", fastAvalancheCalculation, true);
  sim.get("fastPercolationCalculation", fastPercolationCalculation, false);
  sim.get("homogeneousGM", homogeneousGM, false);
  sim.get("runFindPercolationPoint", runFindPercolationPoint, false);
  sim.get("xInitialStep", xInitialStep, 1.0);
  sim.get("xMinimumStep", xMinimumStep, 0.1);
  sim.get("percolationThreshold", percolationThreshold, 1.0);

  sim.get("depression", depression, false);
  if(depression)
  {
    sim.get("beta", beta, 0.9);
    sim.get("tauD", tau_D, 50);
  }

  string tmpstr;
  sim.get("model", tmpstr, "GM");
  if(tmpstr == "GM")
    model = MODEL_GM;
  else if(tmpstr == "BP")
    model = MODEL_BP;
  else if(tmpstr == "RNM")
    model = MODEL_RNM;
  else if(tmpstr == "RNMBP")
    model = MODEL_RNMBP;
  else if(tmpstr == "GMP0")
    model = MODEL_GMP0;
  else if(tmpstr == "REALBP")
    model = MODEL_REALBP;
  else if(tmpstr == "REALBPS")
    model = MODEL_REALBPS;
  else if(tmpstr == "RNMREALBPS")
    model = MODEL_RNMREALBPS;
  else
  {
    model = MODEL_GM;
    cout << "Warning: model not well defined." << endl
         << "Current possibilities: GM, GMP0, BP and RNM. Using GM..." << endl;
  }

  xCurrentStep = xInitialStep;

  // Initialize the RNG
  rngSeed = initializeRNG();

  // Load the network connectivity matrix
  loadNetwork();

  // Initialize other variables

  // Number of active inputs (used to calculate the probabilities)
  activeInputs = VectorXd::Zero(N);

  // Means calculations
  if(calculateMeans)
  {
    // Mean number of active neurons
    meanActivity = VectorXd::Zero(steps);
    // Firing probability
    meanP = VectorXd::Zero(steps);
    stdP = VectorXd::Zero(steps);
    if(depression)
      meanD = VectorXd::Zero(steps);

    // Branching parameter
    meanBranching = VectorXd::Zero(steps);
    meanBranching2 = VectorXd::Zero(steps);
    stdBranching = VectorXd::Zero(steps);
  }

  // Current and previous neuron state
  S = VectorXd::Zero(N);
  Sprev = VectorXd::Zero(N);

  // Current and previous spike index - For avalanches only
  if(calculateAvalanches)
  {
    I = VectorXd::Zero(N);
    Iprev = VectorXd::Zero(N);
    inputVector = VectorXd::Zero(N);
  }

  // For synaptic depression
  if(depression)
  {
    D = VectorXd::Ones(N);
    cout << "Short term synaptic depression is ON" << endl;
  }
  else
  {
    cout << "Short term synaptic depression is OFF" << endl;
  }

  percolationEvents = 0;
  // Set the real BP probabilities
  if(model == MODEL_REALBP || model == MODEL_REALBPS)
  {
    cout << "Setting P_m(0) = " << p0 << ", P_m(1) = " << p1 << ", m0 = " << Kmax << " for the real branching process..." << endl;
    m0 = Kmax+1;
  }
  if(model == MODEL_RNMREALBPS)
  {
    cout << "Setting P_m(0) = " << p0 << ", P_m(1) = " << p1 << ", m0 = " << Kmax << " for the RNM real branching process..." << endl;
    m0 = Kmax+1;
  }
  // Precompute the Normalized Complementary Incomplete Gamma Function
  precomputeGammaFunction();

  // If we are using the BP model set p0 to 0
  if(model == MODEL_BP || model == MODEL_RNMBP)
  {
    cout << "Setting P_m(0) = 0 for the branching process..." << endl;
    P_m(0) = 0;
  }
  if(model == MODEL_GMP0)
  {
    cout << "Setting P_m(0) = " << p0 << " for the GMP0 model..." << endl;
    P_m(0) = p0;
    if(p0 == 0)
    {
      cout << "P_M(0)=0. Defaulting to the branching process model..." << endl;
      model = MODEL_BP;
    }
  }
  // Open continous output streams
  string ofile;

  ofile  = outputBaseFile + outputPercolationSteps + outputExtension;
  outputPercolationStepsStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
  outputPercolationStepsStream.precision(5);

  if(saveSpikesData)
  {
    ofile = outputBaseFile + outputSpikes + outputExtension;
    outputSpikesStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
    outputSpikesStream.precision(5);
  }

  if(calculateAvalanches)
  {
    ofile = outputBaseFile + outputAvalancheStatistics + outputExtension;
    outputAvalancheStatisticsStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
    outputAvalancheStatisticsStream.precision(5);
    if(!fastAvalancheCalculation)
    {
      ofile = outputBaseFile + outputAvalancheStatistics + "Slow" + outputExtension;
      outputAvalancheStatisticsSlowStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
      outputAvalancheStatisticsSlowStream.precision(5);
    }
    if(saveFullAvalancheData)
    {      
      ofile = outputBaseFile + outputAvalancheStatistics + "Full" + outputExtension;
      outputAvalancheStatisticsFullStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
      outputAvalancheStatisticsFullStream.precision(5);

      ofile = outputBaseFile + outputAvalancheStatistics + "ParentRelations" + outputExtension;
      outputParentRelationsStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
      outputParentRelationsStream.precision(5);
    }
  }
  if(runFindPercolationPoint)
  {
    ofile = outputBaseFile + outputPercolationPoint + outputExtension;
    outputPercolationPointStream.open(ofile.c_str(), ofstream::out | ofstream::trunc);
    outputPercolationPointStream.precision(10);
  }

  // Preallocate spikes vectors
  totalSpikeCount = 0;
  currentSpike = 0;
  currentLink = 0;
  currentParentRelation = 0;
  spikeNeuron = VectorXi::Zero(maxVectorSize);
  spikeTime = VectorXi::Zero(maxVectorSize);
  spikeIndex = VectorXi::Zero(maxVectorSize);
  spikeInputs = VectorXi::Zero(maxVectorSize);
  spikeParent = VectorXi::Zero(maxVectorSize);
  spikeGlobalIndex = VectorXi::Zero(maxVectorSize);
  spikeClusterSize = VectorXi::Zero(maxVectorSize);
  spikeClusterInputs = VectorXi::Zero(maxVectorSize);
  spikeClusterDuration = ArrayXXi::Zero(maxVectorSize, 2);
  if(saveFullAvalancheData)
  {
    avalancheLinksData = ArrayXXi::Zero(maxVectorSize, 5);
    avalancheParentRelations = ArrayXXi::Zero(maxVectorSize, 2);
  }
  prevStepClean = false;
}

void Kernel::resetSimulation()
{
  Sprev.setZero();
  resetSpikes();

  totalSpikeCount = 0;
  precomputeGammaFunction(false);
}


void Kernel::precomputeGammaFunction(bool verbose)
{
  double vsize = std::max(Kmax, double(m0));
  P_m = VectorXd::Ones(vsize+1);
  for(int m = 0; m < m0; m++)
    P_m(m) = gsl_sf_gamma_inc_P(m0-m, x);

  // Overwrite the P_m for the branching model
  if(model == MODEL_REALBP)
  {
    P_m(0) = p0;
    for(int m = 1; m < m0; m++)
      P_m(m) = 1-gsl_sf_pow_int(1-p1, m);
  }
  if(model == MODEL_REALBPS || model == MODEL_RNMREALBPS)
  {
    for(int m = 0; m < m0; m++)
      P_m(m) = 1-(1-p0)*gsl_sf_pow_int(1-p1, m);
  }
  if(homogeneousGM)
  {
    for(int m = 1; m <= vsize; m++)
      P_m(m) = P_m(0);
    if(verbose)
      cout << endl << "Using the homogeneous GM. P_m = P_0 for all m." << endl;
  }
  if(verbose)
  {
    cout << endl << "Gamma function preloaded for x: " << x << " and m0: " << m0 << endl;
    cout << "Values until m0:" << endl;
    cout << P_m.head(m0+1) << endl;
    cout << "Gamma function size: " << vsize << endl;
  }
}

// Save avalanche data / partial only saves half
void Kernel::saveAvalanches(bool partial)
{
  int numSpikesToSave = currentSpike;
  if(partial)
  {
    numSpikesToSave = floor(currentSpike/2);
  }
  // Before saving the spikes I have to go through all and check the parent again
  for(int i = 0; i < numSpikesToSave; i++)
  {
    spikeParent(i) = findRoot(i);
    // If i am not my parent, my cluster size has to be 0
    if(spikeParent(i) != i)
      spikeClusterSize(i) = 0;
  }
  // Fast code. All avalanche should already be stored in the cluster sizes
  // For now just sizes and durations
  for(int i = 0; i < numSpikesToSave; i++)
    if(spikeClusterSize(i) > 0)
      outputAvalancheStatisticsStream << spikeClusterSize(i) << " " << spikeClusterDuration(i, 1)-spikeClusterDuration(i, 0) << " " << spikeClusterInputs(i) << endl;

    // Should be unaffected by partial
  if(saveFullAvalancheData)
  {
    for(int i = 0; i < currentLink; i++)
      outputAvalancheStatisticsFullStream << avalancheLinksData.row(i) << endl;
    for(int i = 0; i < currentParentRelation; i++)
      outputParentRelationsStream << avalancheParentRelations.row(i) << endl;
  }
}
// Save spike data / partial only saves half
void Kernel::saveSpikes(bool partial)
{
  int numSpikesToSave = currentSpike;
  if(partial)
  {
    numSpikesToSave = floor(currentSpike/2);
  }
  for(int i = 0; i < numSpikesToSave; i++)
  {
    outputSpikesStream << spikeNeuron(i) << " " << spikeTime(i) << " "
                       << spikeIndex(i) << " " << spikeParent(i) + spikeGlobalIndex(0)
           << " " << spikeGlobalIndex(i) << " " << spikeClusterSize(i) << " " << spikeClusterDuration(i, 1)-spikeClusterDuration(i, 0) << " " << spikeInputs(i) << " " << spikeClusterInputs(i) << "\n";
  }
}

// Reset spike data / partial only saves half
void Kernel::resetSpikes(bool partial)
{
  if(partial)
  {
    /*std::cout << "Before the reset: " << spikeNeuron(currentSpike-1) << " " << spikeTime(currentSpike-1) << " "
                       << spikeIndex(currentSpike-1) << " " << spikeParent(currentSpike-1) 
           << " " << spikeGlobalIndex(currentSpike-1) << " " << spikeClusterSize(currentSpike-1) << " " << spikeClusterDuration(currentSpike-1, 1)-spikeClusterDuration(currentSpike-1, 0) << " " << spikeInputs(currentSpike-1) << " " << spikeClusterInputs(currentSpike-1) << "\n";
    std::cout << "Doing a partial reset...";*/

    int numSpikesToSave = floor(currentSpike/2);

    // Make size to store the info not saved
    int numUnsavedSpikes = currentSpike - numSpikesToSave;
    VectorXi tmpVector(numUnsavedSpikes);
    ArrayXXi tmpVector2(numUnsavedSpikes, 2);
    
    tmpVector = spikeNeuron.segment(numSpikesToSave, numUnsavedSpikes);
    spikeNeuron.setZero();
    spikeNeuron.head(numUnsavedSpikes) = tmpVector;

    tmpVector = spikeTime.segment(numSpikesToSave, numUnsavedSpikes);
    spikeTime.setZero();
    spikeTime.head(numUnsavedSpikes) = tmpVector;

    tmpVector = spikeIndex.segment(numSpikesToSave, numUnsavedSpikes);
    spikeIndex.setZero();
    spikeIndex.head(numUnsavedSpikes) = tmpVector;

    tmpVector = spikeInputs.segment(numSpikesToSave, numUnsavedSpikes);
    spikeInputs.setZero();
    spikeInputs.head(numUnsavedSpikes) = tmpVector;

    tmpVector = spikeParent.segment(numSpikesToSave, numUnsavedSpikes);
    spikeParent.setZero();
    spikeParent.head(numUnsavedSpikes) = tmpVector;

    // Fix the parents and spike indices
    for(int i = 0; i < numUnsavedSpikes; i++)
    {
      spikeIndex(i) = spikeIndex(i) - numSpikesToSave;
      if(spikeParent(i) != 0)
      {
        spikeParent(i) = spikeParent(i) - numSpikesToSave;
      }
    }

    // Fix the previous state
    if(calculateAvalanches)
    {
      for(int i = 0; i < N; i++)
      {
        if(Iprev(i) != 0)
        {
          Iprev(i) -= numSpikesToSave;
        }
      }
    }

    tmpVector = spikeGlobalIndex.segment(numSpikesToSave, numUnsavedSpikes);
    spikeGlobalIndex.setZero();
    spikeGlobalIndex.head(numUnsavedSpikes) = tmpVector;

    tmpVector = spikeClusterSize.segment(numSpikesToSave, numUnsavedSpikes);
    spikeClusterSize.setZero();
    spikeClusterSize.head(numUnsavedSpikes) = tmpVector;

    tmpVector = spikeClusterInputs.segment(numSpikesToSave, numUnsavedSpikes);
    spikeClusterInputs.setZero();
    spikeClusterInputs.head(numUnsavedSpikes) = tmpVector;

    tmpVector2 = spikeClusterDuration.middleRows(numSpikesToSave, numUnsavedSpikes);
    spikeClusterDuration.setZero();
    spikeClusterDuration.topRows(numUnsavedSpikes) = tmpVector2;

    currentSpike = numUnsavedSpikes;
    
  }
  else
  {
    spikeNeuron.setZero();
    spikeTime.setZero();
    spikeIndex.setZero();
    spikeInputs.setZero();
    spikeParent.setZero();
    spikeGlobalIndex.setZero();
    spikeClusterSize.setZero();
    spikeClusterInputs.setZero();
    spikeClusterDuration.setZero();
    
    currentSpike = 0;
  }

  // Should be unaffected by patial saves
  if(saveFullAvalancheData)
  {
    avalancheLinksData.setZero();
    avalancheParentRelations.setZero();
    currentLink = 0;
    currentParentRelation = 0;
  }
}

void Kernel::loadNetwork()
{
  cout << endl << "Loading the network..." << endl;
  // Load the network and generate the sparse connectivity matrix
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;

  ifstream linksFile(networkLinksFile.c_str());
  ifstream weightsFile(networkWeightsFile.c_str());
  string linerow, linecol;
  int row,col, val;
  std::getline(linksFile, linerow);
  std::getline(linksFile, linecol);
  stringstream linerowStream(linerow);
  stringstream linecolStream(linecol);
  while (linerowStream >> row)
  {
    linecolStream >> col;
    weightsFile >> val;
    tripletList.push_back(T(row,col,val));
  }
  linksFile.close();
  weightsFile.close();
  RS = SparseMatrix<double,RowMajor>(N, N);
  RS.setFromTriplets(tripletList.begin(), tripletList.end());

  // To store the output effective network
  if(calculateEffectiveNetwork)
  {
//E    RSeffectiveOriginal = sp_mat(conmat, conv_to<vec>::from(vals), networkSize, networkSize);
//E    RSeffective = RSeffectiveOriginal;
  }

  RST = RS.transpose();
  // Calculate maximum input connectivity (columns in RS)
  Kmax = 0;
  double tmpK;
  for (int k=0; k<RS.outerSize(); ++k)
  {
    tmpK = 0;
    for (SparseMatrix<double>::InnerIterator it(RS,k); it; ++it)
    {
      tmpK += it.value();
    }
    Kmax = max(tmpK, Kmax);
  }

  //  cout << RS << endl;
  cout << "Connectivity Matrix loaded." << endl
       << "---------------------------" << endl
       << "Links file: " << networkLinksFile << endl
       << "Weights file: " << networkWeightsFile << endl
       << "Size: " << networkSize << "x" << networkSize << endl
       << "Elements: " << RS.nonZeros() << endl
       << "Kmax: " << Kmax << endl
       << "---------------------------" << endl << endl;
}

int Kernel::initializeRNG()
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  struct tm *tm = localtime(&tv.tv_sec);
  int seed = abs(int(tv.tv_usec/10+tv.tv_sec*100000));  // Creates the seed based on actual time
  rng = gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set(rng,seed);      // Seeds the previously created RNG

  cout << endl << "RNG initialized. Seed: " << seed << endl
       << "Current date: " << tm->tm_mday << "/" << tm->tm_mon +1 << "/" << tm->tm_year + 1900 << endl
       << "Current time: " << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec << endl;

  return seed;
}

void Kernel::execute(Sim& sim)
{
  int NpartialSteps = 10000;
  // Start counting time
  time(&start);
  cout << endl << "Starting the simulation..." << endl;
  // Special mode
  bool done = false;
  if(runFindPercolationPoint)
  {
    while(!done)
    {
      for(currentStep = 0; currentStep < steps; currentStep++)
      {
        if(step())
        {
          // Percolation achieved. Simply decrease x and reset the system
          x -= xCurrentStep;
          while( x <= 0)
          {
            x += xCurrentStep;
            xCurrentStep = xCurrentStep/2.0;
            if(xCurrentStep < xMinimumStep)
            {
              done = true;
              break;
            }
            else
              x -= xCurrentStep;
          }
          resetSimulation();
          break; // out of the for and into the while
        }
      }
      // If I got here, percolation was not achieved. Go back and decrease step size
      if(currentStep == steps)
      {
        cout << "Percolation not reached for x = " << x << endl;
        // Percolation not achieved. Go back and decrease step size
        x += xCurrentStep;
        xCurrentStep = xCurrentStep/2.0;
        if(xCurrentStep < xMinimumStep)
        {
          // Minimum step achieved. Exit
          done = true;
        }
        // Decrease step size
        else
        {
          x -= xCurrentStep;
          while( x <= 0)
          {
            x += xCurrentStep;
            xCurrentStep = xCurrentStep/2.0;
            if(xCurrentStep < xMinimumStep)
            {
              done = true;
              break;
            }
            else
              x -= xCurrentStep;
          }
          resetSimulation();
        }
      }
    }
    return;
  }

  // Normal mode
  for(currentStep = 0; currentStep < steps; currentStep++)
  {
    // if step return 1 finish. If 0 continue
    if(step())
      return;
    if(currentStep == NpartialSteps-1)
    {
      time(&partial);
      cout << "Time elapsed after " << long(NpartialSteps) << " steps: " << sec2string(difftime(partial,start)) << endl;
      cout << "ETA: " << sec2string(difftime(partial,start)*(steps-double(NpartialSteps))/double(NpartialSteps)) << endl; 
    }
  }
}

int Kernel::step()
{
  int val;
  switch(model)
  {
    case MODEL_GM:
      val = stepGM();
      break;
    case MODEL_BP:
      val = stepGM();
      break;
    case MODEL_RNM:
      val = stepGM();
      break;
    case MODEL_RNMBP:
      val = stepGM();
      break;
    case MODEL_GMP0:
      val = stepGM();
      break;
    case MODEL_REALBP:
      val = stepGM();
      break;
    case MODEL_REALBPS:
      val = stepGM();
    case MODEL_RNMREALBPS:
      val = stepGM();
      break;
    default:
      val = stepGM();
  }
  return val;
}

// Simulation step for the GM model
int Kernel::stepGM()
{
  // RNM SPECIFIC
  // Shuffle the previous state before computing anything
  if(model == MODEL_RNM || model == MODEL_RNMBP || model == MODEL_RNMREALBPS)
  {
    double *c_ptr;
    if(calculateAvalanches)
    {
      // Clone the rng
      gsl_rng* tmpRng = gsl_rng_clone(rng); 
      c_ptr = Iprev.data();
      gsl_ran_shuffle(tmpRng, c_ptr, N, sizeof(int));
      Iprev = Map<VectorXd>(c_ptr, N);
    }
    c_ptr = Sprev.data();
    gsl_ran_shuffle(rng, c_ptr, N, sizeof(int));
    Sprev = Map<VectorXd>(c_ptr, N);

  }
  // Calculate the active inputs
  if(depression)
    activeInputs = RST*(D.cwiseProduct(Sprev));
  else
    activeInputs = RST*Sprev;

  // Calculate the firing probabilities
  VectorXd P(N);
  for(int i = 0; i < N; i++)
  {
    if(depression)
    {
      P(i) = (P_m((int)floor(activeInputs(i)+1))-P_m((int)floor(activeInputs(i))))*(activeInputs(i)-floor(activeInputs(i)))+P_m((int)floor(activeInputs(i)));
    }
    else
      P(i) = P_m((int)round(activeInputs(i)));
  }
  // Reset current state
  S.setZero();
  if(calculateAvalanches)
  {
    I.setZero();
  }
  // BP SPECIFIC
  // if there are no active inputs activate one neuron at random (better, set its firing probability to 1)
  // The avalancheOnlyHack is for when we only compute avalanche statistics and we don't want to wait for a random activitation in case p0 is very very small
  if(((model == MODEL_BP || model == MODEL_RNMBP || model == MODEL_REALBP || model == MODEL_REALBPS || model == MODEL_RNMREALBPS) && P_m(0) == 0 && activeInputs.sum() == 0) || (avalancheOnlyHack && activeInputs.sum() == 0))
  {
    // We need one empty step
    if(prevStepClean)
    {
      //S(gsl_rng_uniform_int(rng, N)) = 1;
      P(gsl_rng_uniform_int(rng, N)) = 1;
      Iprev.setZero(); // Just in case
      prevStepClean = false;
    }
    else
      prevStepClean = true;
  }

  // Check if they fire
  for(int i = 0; i < N; i++)
  {
    if(P(i) > 0 && gsl_rng_uniform(rng) < P(i))
    {
      // There's a new spike
      S(i) = 1;
      // Update depression
      if(depression)
      {
        D(i) *= D(i)*beta;
      }
      spikeNeuron(currentSpike) = i;
      spikeTime(currentSpike) = currentStep;
      spikeIndex(currentSpike) = currentSpike;
      spikeInputs(currentSpike) = activeInputs(i);
      spikeParent(currentSpike) = currentSpike;
      spikeGlobalIndex(currentSpike) = totalSpikeCount;
      spikeClusterSize(currentSpike) = 1;

      spikeClusterInputs(currentSpike) = activeInputs(i);

      spikeClusterDuration(currentSpike, 0) = currentStep;
      spikeClusterDuration(currentSpike, 1) = currentStep;

      if(calculateAvalanches)
      {
        int ret = stepCalculateAvalanches(i);
        if(ret == 1)
          return 1;
      }

      currentSpike++;
      totalSpikeCount++;

    }
  }
  // Do the evolution of depression
  if(depression)
  {
    D += 1/tau_D*(VectorXd::Ones(N)-D); // Should be a vector operation
  }
  // Calculate the means
  if(calculateMeans)
  {
    VectorXd tmpVector;
    double *c_ptr;

    tmpVector = S.cast<double>();
    c_ptr = tmpVector.data();
    meanActivity(currentStep) = gsl_stats_mean(c_ptr, 1, N);

    tmpVector = P.cast<double>();
    c_ptr = tmpVector.data();
    meanP(currentStep) = gsl_stats_mean(c_ptr, 1, N);
    stdP(currentStep) = gsl_stats_sd(c_ptr, 1, N);
    if(depression)
    {
      //double *d_ptr;
      //tmpVector = D.cast<double>(); // Probably don't need all thoses castas since they are doubles now
      c_ptr = D.data();
      meanD(currentStep) = gsl_stats_mean(c_ptr, 1, N);
    }

    currentBranching = Sprev.cwiseProduct(RS*S);
    double currActive = Sprev.sum();
    tmpVector = currentBranching.cast<double>();
    c_ptr = tmpVector.data();
    // Fix so we only calculate the branching parameter if the neuron was firing
    if(currActive > 1)
    {
      double mb = tmpVector.sum()/currActive;
      meanBranching(currentStep) = mb;
      
      //stdBranching(currentStep) = gsl_stats_variance_m(c_ptr, 1, N, mb);
      stdBranching(currentStep) = (1/(currActive-1))*
                                  (gsl_stats_tss_m(c_ptr, 1, N, mb)-(N-currActive)*mb*mb);

      meanBranching2(currentStep) = tmpVector.sum()/double(S.sum());
    }
    else if(currActive == 1)
    {
      meanBranching(currentStep) = tmpVector.sum();
      stdBranching(currentStep) = 0;      
      meanBranching2(currentStep) = tmpVector.sum();
    }
    else
    {
      meanBranching(currentStep) = 0;
      stdBranching(currentStep) = 0;
      meanBranching2(currentStep) = 0;
    }
  }

  // Shift the state
  Sprev = S;
  if(calculateAvalanches)
    Iprev = I;

  // Check for percolation and reset the system
  if(Sprev.sum() >= N*percolationThreshold && !depression)
  {
    cout << "Percolation reached";
    Sprev.setZero();
    percolationEvents++;
    outputPercolationStepsStream << currentStep << endl;
    if(calculateAvalanches)
    {
      Iprev.setZero();
      saveAvalanches();
    }
    if(saveSpikesData)
    {
      saveSpikes();
    }
    resetSpikes();
    if(runFindPercolationPoint)
    {
      cout << " for x = " << x << endl;
      // Store the time and x value for reference and return
      outputPercolationPointStream << currentStep << " " << x << endl;
      return 1;
    }
    cout << endl;
    if(fastPercolationCalculation)
    {
      cout << "fastPercolationCalculation is on. Quitting." << endl;
      return 1;
    }
  }

  // Vector limit checks so we never fill it
  if(currentSpike >= maxVectorSize*0.9 || currentLink >= maxVectorSize*0.9 && S.sum() != 0)
  {
    cout << "Warning: Limit vector size reached at fraction: " << double(currentStep)/double(steps) << endl;
    if(calculateAvalanches)
    {
      saveAvalanches(true);
    }
    if(saveSpikesData)
    {
      saveSpikes(true);
    }
    resetSpikes(true);
  }

  // Also, if no spike fired this step, we can safely store everything
  if(S.sum() == 0 && currentSpike > 10000)
  {
    if(calculateAvalanches)
    {
      Iprev.setZero();
      saveAvalanches();
    }
    if(saveSpikesData)
    {
      saveSpikes();
    }
    resetSpikes();
  }
  return 0;
}
// There's a bug here somewhere
int Kernel::stepCalculateAvalanches(int activeNeuron)
{
  int i = activeNeuron;
  I(i) = currentSpike;
  // I only need to iterate over the nonzero entries of the ith column 
  // of the RS matrix if I have more than 1 active input
  if(activeInputs(i) > 0)
  {
    for (SparseMatrix<double>::InnerIterator it(RS,i); it; ++it)
    {
      int j = it.row();
      // I'm only iterating over the non-zero entries in the RS column
      // (inputs), so j is already an input connection. 
      // Now I only have to check if it was active the previous time step
      if(Iprev(j) > 0)
      {
        // Quite expensive operation here
        if(calculateEffectiveNetwork)
          RSeffective.coeffRef(j,i)++;

        parent = findRoot(Iprev(j));
        // If the parent is lower I (me and my whole cluster) go to him
        if(parent < spikeParent(currentSpike) && parent >= 0)
        {
          if(saveFullAvalancheData)
          {
            avalancheParentRelations(currentParentRelation, 0) = parent+spikeGlobalIndex(0);
            avalancheParentRelations(currentParentRelation, 1) = spikeParent(currentSpike)+spikeGlobalIndex(0);
            currentParentRelation++;
          }
          // Add the spike to the parent
          spikeClusterSize(parent) += spikeClusterSize(spikeParent(currentSpike));
          spikeClusterSize(spikeParent(currentSpike)) = 0;

          spikeClusterInputs(parent) += spikeClusterInputs(spikeParent(currentSpike));
          spikeClusterInputs(spikeParent(currentSpike)) = 0;

          spikeClusterDuration(parent, 0) = min(spikeClusterDuration(parent, 0), spikeClusterDuration(spikeParent(currentSpike), 0));
          spikeClusterDuration(parent, 1) = max(spikeClusterDuration(parent, 1), spikeClusterDuration(spikeParent(currentSpike), 1));

          spikeParent(spikeParent(currentSpike)) = parent;
          spikeParent(currentSpike) = parent;

        }
        // If the parent is higher he goes to my parent
        else if(parent > spikeParent(currentSpike))
        {              
          if(saveFullAvalancheData)
          {
            avalancheParentRelations(currentParentRelation, 0) = spikeParent(currentSpike)+spikeGlobalIndex(0);
            avalancheParentRelations(currentParentRelation, 1) = parent+spikeGlobalIndex(0);
            currentParentRelation++;
          }
          // Add the whole cluster size to the parent
          spikeClusterSize(spikeParent(currentSpike)) += spikeClusterSize(parent);
          spikeClusterSize(parent) = 0;
          //spikeClusterSize(currentSpike) = 0;

          spikeClusterInputs(spikeParent(currentSpike)) += spikeClusterInputs(parent);
          spikeClusterInputs(parent) = 0;

          spikeClusterDuration(parent, 0) = min(spikeClusterDuration(parent, 0), spikeClusterDuration(spikeParent(currentSpike), 0));
          spikeClusterDuration(parent, 1) = max(spikeClusterDuration(parent, 1), spikeClusterDuration(spikeParent(currentSpike), 1));

          spikeParent(parent) = spikeParent(currentSpike);
        }
        else if( parent < 0)
        {
          std::cout << "Error! Found a negative parent, avalanches might be too big, try increasing the maxVectorSize\n";
          //exit(1);
          return 1;
        }
        // If the parent is the same, there's nothing to be done
        else
        {

        }

        // For the j connections that were active, store the links
        if(saveFullAvalancheData)
        {
          avalancheLinksData(currentLink, 0) = spikeParent(currentSpike);
          avalancheLinksData(currentLink, 1) = j;
          avalancheLinksData(currentLink, 2) = i;
          avalancheLinksData(currentLink, 3) = currentStep-1;
          avalancheLinksData(currentLink, 4) = spikeGlobalIndex(currentSpike);
          currentLink++;
        }
      }
    }
  }
  else
  {
    if(saveFullAvalancheData)
    {
      avalancheLinksData(currentLink, 0) = spikeParent(currentSpike);
      avalancheLinksData(currentLink, 1) = i;
      avalancheLinksData(currentLink, 2) = i;
      avalancheLinksData(currentLink, 3) = currentStep;
      avalancheLinksData(currentLink, 4) = spikeGlobalIndex(currentSpike);
      currentLink++;
    }
  }
  return 0;
}

void Kernel::finalize(Sim& sim)
{

  time(&end);
  cout << endl << "Simulation finished." << endl;
  cout << "Total time elapsed: " << sec2string(difftime(end,start)) << endl;
  cout << "Mean Percolation time: " << double(steps)/double(percolationEvents) << endl;
  cout << endl << "Storing output..." << endl;

  // Save the output
  ofstream ofs;
  string ofile;

  // Save the means
  if(calculateMeans)
  // Save mean Activity
  {
    ofile = outputBaseFile + outputMeans + outputExtension;
    ofs.open(ofile.c_str(), ofstream::out | ofstream::trunc);
    ofs.precision(10);
    for (int i = 0; i < currentStep; i++)
    {
      ofs << meanActivity(i) << " " << meanP(i) << " " << stdP(i) << " " << meanBranching(i) << " " << stdBranching(i) << " " << meanBranching2(i);
      if(depression)
        ofs << " " << meanD(i);
      ofs << endl;
    }
    ofs.close();
  }

  // Close continuous output streams
  outputPercolationStepsStream.close();

  if(calculateAvalanches)
  {
    saveAvalanches();
    outputAvalancheStatisticsStream.close();
    if(!fastAvalancheCalculation)
      outputAvalancheStatisticsSlowStream.close();
    if(saveFullAvalancheData)
    {
      outputAvalancheStatisticsFullStream.close();
      outputParentRelationsStream.close();
    }
  }
 
  if(saveSpikesData)
  {
    saveSpikes();
    outputSpikesStream.close();
  }
  resetSpikes();

  if(calculateEffectiveNetwork)
  {
  //E  sp_mat tmat = RSeffective-RSeffectiveOriginal;
  //E  saveSparse(tmat, networkOutputLinksFile, networkOutputWeightsFile);
  }

  if(runFindPercolationPoint)
  {
    outputPercolationPointStream.close();
  }

  cout << "Done." << endl;
}

int Kernel::findRoot(int targetSpike)
{
  if(targetSpike < 0)
    return targetSpike;
  if (spikeParent(targetSpike) == targetSpike)
    return targetSpike;

  return spikeParent(targetSpike) = findRoot(spikeParent(targetSpike));
}

/*
void Kernel::printSparse(sp_umat M)
{
  unsigned int L = M.n_nonzero;
  unsigned int N = M.n_cols;
  unsigned int fv,lv;
  for(int i = 0; i < N; i++)
  {
    fv = M.col_ptrs[i];
    if(i < N-1)
      lv = M.col_ptrs[i+1]-1;
    else
      lv = L-1;
    for(int j = fv; j <= lv; j++)
      cout << M.row_indices[j] << " " << i << " " << M.values[j] << endl;
  }
}

void Kernel::saveSparse(sp_umat M, string linksFile, string weightsFile)
{
  ofstream ofs;

  unsigned int L = M.n_nonzero;
  unsigned int N = M.n_cols;
  unsigned int fv,lv;

  urowvec r(L);
  urowvec c(L);
  vec w(L);

  unsigned int k = 0;
  for(int i = 0; i < N; i++)
  {
    fv = M.col_ptrs[i];
    if(i < N-1)
      lv = M.col_ptrs[i+1]-1;
    else
      lv = L-1;
    for(int j = fv; j <= lv; j++)
    {
      r(k) = M.row_indices[j];
      c(k) = i;
      w(k) = M.values[j];
      k++;
    }
  }

  // Save the information
  ofs.open(linksFile.c_str(), ofstream::out | ofstream::trunc);
  r.save(ofs, raw_ascii);
  c.save(ofs, raw_ascii);
  ofs.close();

  ofs.open(weightsFile.c_str(), ofstream::out | ofstream::trunc);
  ofs.precision(5);
  w.raw_print(ofs);
  ofs.close();
}

void Kernel::saveSparse(sp_mat M, string linksFile, string weightsFile)
{
  ofstream ofs;

  unsigned int L = M.n_nonzero;
  unsigned int N = M.n_cols;
  unsigned int fv,lv;

  urowvec r(L);
  urowvec c(L);
  vec w(L);
  //cout << L << " " << N << endl;
  unsigned int k = 0;
  for(int i = 0; i < N; i++)
  {
    fv = M.col_ptrs[i];
    if(i < N-1)
      lv = M.col_ptrs[i+1]-1;
    else
      lv = L-1;
    for(int j = fv; j <= lv; j++)
    {
      r(k) = M.row_indices[j];
      c(k) = i;
      w(k) = M.values[j];
      k++;
    }
  }

  // Save the information
  ofs.open(linksFile.c_str(), ofstream::out | ofstream::trunc);
  r.save(ofs, raw_ascii);
  c.save(ofs, raw_ascii);
  ofs.close();

  ofs.open(weightsFile.c_str(), ofstream::out | ofstream::trunc);
  ofs.precision(5);
  w.raw_print(ofs);
  ofs.close();
}*/

// Auxilary functions, taken from Olav
// https://github.com/olavolav/te-causality
std::string sec2string(double seconds) {
  if(seconds > double(std::numeric_limits<long>::max())) return "inf";
  if(seconds < double(std::numeric_limits<long>::min())) return "-inf";
  return sec2string((long)seconds);
}
std::string sec2string(long seconds)
{
  std::ostringstream text(std::ostringstream::out);

  if(seconds<0) {
    text <<"-";
    seconds *= -1;
  }

  if(seconds>3600*24) {
    text << seconds/(3600*24) << "d ";
    seconds = seconds % (3600*24);
  }

  if(seconds>3600) {
    text << seconds/3600 << "h ";
    seconds = seconds % 3600;
  }

  if(seconds>60) {
    text << seconds/60 << "m ";
    seconds = seconds % 60;
  }
  text << seconds << "s";

  return text.str();
}
