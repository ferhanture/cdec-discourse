#include <sstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "config.h"

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include "sentence_metadata.h"
#include "scorer.h"
#include "verbose.h"
#include "viterbi.h"
#include "hg.h"
#include "prob.h"
#include "kbest.h"
#include "ff_register.h"
#include "decoder.h"
#include "filelib.h"
#include "fdict.h"
#include "time.h"
#include "sampler.h"

#include "weights.h"
#include "sparse_vector.h"

using namespace std;
using boost::shared_ptr;
namespace po = boost::program_options;

//@author FERHANTURE
int get_doc_id(string s);
int get_sent_id(string s);
vector<string> split(const string& s, const string& delim, const bool keep_empty);


bool invert_score;
boost::shared_ptr<MT19937> rng;
bool approx_score;
bool no_reweight;
bool no_select;
bool unique_kbest;
int update_list_size;
vector<double> dense_weights;
double mt_metric_scale;
int optimizer;

void SanityCheck(const vector<double>& w) {
    for (int i = 0; i < w.size(); ++i) {
        assert(!isnan(w[i]));
        assert(!isinf(w[i]));
    }
}

struct FComp {
    const vector<double>& w_;
    FComp(const vector<double>& w) : w_(w) {}
    bool operator()(int a, int b) const {
        return fabs(w_[a]) > fabs(w_[b]);
    }
};

void ShowLargestFeatures(const vector<double>& w) {
    vector<int> fnums(w.size());
    for (int i = 0; i < w.size(); ++i)
        fnums[i] = i;
    vector<int>::iterator mid = fnums.begin();
    mid += (w.size() > 10 ? 10 : w.size());
    partial_sort(fnums.begin(), mid, fnums.end(), FComp(w));
    cerr << "TOP FEATURES:";
    for (vector<int>::iterator i = fnums.begin(); i != mid; ++i) {
        cerr << ' ' << FD::Convert(*i) << '=' << w[*i];
    }
    cerr << endl;
}

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
    po::options_description opts("Configuration options");
    opts.add_options()
    ("input_weights,w",po::value<string>(),"Input feature weights file")
    ("source,i",po::value<string>(),"Source file for development set")
    ("passes,p", po::value<int>()->default_value(15), "Number of passes through the training data")
    ("reference,r",po::value<vector<string> >(), "[REQD] Reference translation(s) (tokenized text file)")
    ("mt_metric,m",po::value<string>()->default_value("ibm_bleu"), "Scoring metric (ibm_bleu, nist_bleu, koehn_bleu, ter, combi)")
    ("optimizer,o",po::value<int>()->default_value(1), "Optimizer (sgd=1, mira 1-fear=2, full mira w/ cutting plane=3, full mira w/ nbest list=5, local update=4)")
    ("max_step_size,C", po::value<double>()->default_value(0.01), "regularization strength (C)")
    ("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
    ("mt_metric_scale,s", po::value<double>()->default_value(1.0), "Amount to scale MT loss function by")
    ("approx_score,a", "Use smoothed sentence-level BLEU score for approximate scoring")
    ("no_reweight,d","Do not reweight forest for cutting plane")
    ("no_select,n", "Do not use selection heuristic")
    ("k_best_size,k", po::value<int>()->default_value(250), "Size of hypothesis list to search for oracles")
    ("update_k_best,b", po::value<int>()->default_value(1), "Size of good, bad lists to perform update with")
    ("unique_k_best,u", "Unique k-best translation list")
    ("weights_output,O",po::value<string>(),"Directory to write weights to")
    ("output_dir,D",po::value<string>(),"Directory to place output in")
    ("decoder_config,c",po::value<string>(),"Decoder configuration file");
    po::options_description clo("Command line options");
    clo.add_options()
    ("config", po::value<string>(), "Configuration file")
    ("help,h", "Print this help message and exit");
    po::options_description dconfig_options, dcmdline_options;
    dconfig_options.add(opts);
    dcmdline_options.add(opts).add(clo);
    
    po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
    if (conf->count("config")) {
        ifstream config((*conf)["config"].as<string>().c_str());
        po::store(po::parse_config_file(config, dconfig_options), *conf);
    }
    po::notify(*conf);
    
    if (conf->count("help") || !conf->count("input_weights") || !conf->count("decoder_config") || !conf->count("reference")) {
        cerr << dcmdline_options << endl;
        return false;
    }
    return true;
}

//load previous translation, store array of each sentences score, subtract it from current sentence and replace with new translation score


static const double kMINUS_EPSILON = -1e-6;
static const double EPSILON = 0.000001;
static const double SMO_EPSILON = 0.0001;
static const double PSEUDO_SCALE = 0.9;
static const int MAX_SMO = 100;
int cur_pass;

struct HypothesisInfo {
    SparseVector<double> features;
    vector<WordID> hyp;
    double mt_metric;
    double hope;
    double fear;
    double alpha;
    double oracle_loss;
    SparseVector<double> oracle_feat_diff;
    
};

bool ApproxEqual(double a, double b) {
    if (a == b) return true;
    return (fabs(a-b)/fabs(b)) < EPSILON;
}

typedef shared_ptr<HypothesisInfo> HI;
bool HypothesisCompareB(const HI& h1, const HI& h2 ) 
{
    return h1->mt_metric > h2->mt_metric;
};

bool HopeCompareB(const HI& h1, const HI& h2 ) 
{
    return h1->hope > h2->hope;
};

bool FearCompareB(const HI& h1, const HI& h2 ) 
{
    return h1->fear > h2->fear;
};

bool HypothesisCompareG(const HI& h1, const HI& h2 ) 
{
    return h1->mt_metric < h2->mt_metric;
};


void CuttingPlane(vector<shared_ptr<HypothesisInfo> >* cur_c, bool* again, vector<shared_ptr<HypothesisInfo> >& all_hyp, vector<double> dense_weights)
{
    bool DEBUG_CUT = false;
    shared_ptr<HypothesisInfo> max_fear, max_fear_in_set;
    vector<shared_ptr<HypothesisInfo> >& cur_constraint = *cur_c;
    
    if(no_reweight)
    {
        //find new hope hypothesis
        for(int u=0;u!=all_hyp.size();u++)	
        { 
            double t_score = all_hyp[u]->features.dot(dense_weights);
            all_hyp[u]->hope = 1 * all_hyp[u]->mt_metric + t_score;
            //if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " S:" << t_score << endl; 
            
        }
        
        //sort hyps by hope score
        sort(all_hyp.begin(),all_hyp.end(),HopeCompareB);    
        
        double hope_score = all_hyp[0]->features.dot(dense_weights);
        if(DEBUG_CUT) cerr << "New hope derivation score " << hope_score << endl;
        
        for(int u=0;u!=all_hyp.size();u++)	
        { 
            double t_score = all_hyp[u]->features.dot(dense_weights);
            //all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - hope_score + t_score;
            
            all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - -1*all_hyp[0]->mt_metric - hope_score + t_score; //relative loss
            //      all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric - -1*all_hyp[0]->mt_metric;
            //all_hyp[u]->oracle_feat_diff = all_hyp[0]->features - all_hyp[u]->features;
            //	all_hyp[u]->fear = -1 * all_hyp[u]->mt_metric + t_score;
            //if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " F:" << all_hyp[u]->fear << endl; 
            
        }
        
        sort(all_hyp.begin(),all_hyp.end(),FearCompareB);
        
    }
    //assign maximum fear derivation from all derivations
    max_fear = all_hyp[0];
    
    if(DEBUG_CUT) cerr <<"Cutting Plane Max Fear "<<max_fear->fear ;
    for(int i=0; i < cur_constraint.size();i++) //select maximal violator already in constraint set
    {
        if (!max_fear_in_set || cur_constraint[i]->fear > max_fear_in_set->fear)
            max_fear_in_set = cur_constraint[i];
    }
    if(DEBUG_CUT) cerr << "Max Fear in constraint set " << max_fear_in_set->fear << endl;
    
    if(max_fear->fear > max_fear_in_set->fear + SMO_EPSILON)
    {
        cur_constraint.push_back(max_fear);
        *again = true;
        if(DEBUG_CUT) cerr << "Optimize Again " << *again << endl;
    }
}


double ComputeDelta(vector<shared_ptr<HypothesisInfo> >* cur_p, double max_step_size)
{
    vector<shared_ptr<HypothesisInfo> >& cur_pair = *cur_p;
    double loss = cur_pair[0]->oracle_loss - cur_pair[1]->oracle_loss;
    double margin = -cur_pair[0]->oracle_feat_diff.dot(dense_weights) - -cur_pair[1]->oracle_feat_diff.dot(dense_weights); //TODO: is it a problem that new oracle is used in diff?
    //  double margin = cur_pair[0]->features.dot(dense_weights) - cur_pair[1]->features.dot(dense_weights); //TODO: is it a problem that new oracle is used in diff?
    /*  double num = 
     (cur_pair[0]->oracle_loss - cur_pair[0]->oracle_feat_diff.dot(dense_weights))
     - (cur_pair[1]->oracle_loss - cur_pair[1]->oracle_feat_diff.dot(dense_weights));
     */
    double num = loss - margin;
    SparseVector<double> diff = cur_pair[0]->features;
    diff -= cur_pair[1]->features;
    /*  SparseVector<double> diff = cur_pair[0]->oracle_feat_diff;
     diff -= cur_pair[1]->oracle_feat_diff;*/
    double diffsqnorm = diff.l2norm_sq();
    double delta;
    if (diffsqnorm > 0)
        delta = num / (diffsqnorm * max_step_size);
    else
        delta = 0;
    
    //clip delta (enforce margin constraints)
    delta = max(-cur_pair[0]->alpha, min(delta, cur_pair[1]->alpha));
    return delta;
}


vector<shared_ptr<HypothesisInfo> > SelectPair(vector<shared_ptr<HypothesisInfo> >* cur_c)
{
    bool DEBUG_SELECT= false;
    vector<shared_ptr<HypothesisInfo> >& cur_constraint = *cur_c;
    
    vector<shared_ptr<HypothesisInfo> > pair;
    
    if (no_select){ //skip heuristic search and return oracle and fear for 1-mira
        if(optimizer == 2)
        {
            pair.push_back(cur_constraint[0]);
            pair.push_back(cur_constraint[1]);
            return pair;
        }
    }
    
    for(int u=0;u != cur_constraint.size();u++)	
    {
        shared_ptr<HypothesisInfo> max_fear;
        
        if(DEBUG_SELECT) cerr<< "cur alpha " << u  << " " << cur_constraint[u]->alpha;
        for(int i=0; i < cur_constraint.size();i++) //select maximal violator
        {
            if(i != u)
                if (!max_fear || cur_constraint[i]->fear > max_fear->fear)
                    max_fear = cur_constraint[i];
        }
        if(!max_fear) return pair; //
        
        if(DEBUG_SELECT) cerr << " F" << max_fear->fear << endl;
        
        
        if ((cur_constraint[u]->alpha == 0) && (cur_constraint[u]->fear > max_fear->fear + SMO_EPSILON))
        {
            for(int i=0; i < cur_constraint.size();i++) //select maximal violator
            {
                if(i != u)
                    if (cur_constraint[i]->alpha > 0)
                    {
                        pair.push_back(cur_constraint[u]);
                        pair.push_back(cur_constraint[i]);
                        cerr << "RETJURN from 1" << endl;
                        return pair;
                    }
            }
        }	       
        if ((cur_constraint[u]->alpha > 0) && (cur_constraint[u]->fear < max_fear->fear - SMO_EPSILON))
        {
            for(int i=0; i < cur_constraint.size();i++) //select maximal violator
            {
                if(i != u)	
                    if (cur_constraint[i]->fear > cur_constraint[u]->fear)
                    {
                        pair.push_back(cur_constraint[u]);
                        pair.push_back(cur_constraint[i]);
                        return pair;
                    }
            }  
        }
        
    } 
    return pair; //no more constraints to optimize, we're done here
    
}

struct GoodBadOracle {
    vector<shared_ptr<HypothesisInfo> > good;
    vector<shared_ptr<HypothesisInfo> > bad;
};

struct TrainingObserver : public DecoderObserver {
    TrainingObserver(const int k, const DocScorer& d, vector<GoodBadOracle>* o, vector<ScoreP>* cbs) : ds(d), oracles(*o), corpus_bleu_sent_stats(*cbs), kbest_size(k) {
        // TrainingObserver(const int k, const DocScorer& d, vector<GoodBadOracle>* o) : ds(d), oracles(*o), kbest_size(k) {
        
        //calculate corpus bleu score from previous iterations 1-best for BLEU gain
        if(cur_pass > 0)
        {
            ScoreP acc;
            for (int ii = 0; ii < corpus_bleu_sent_stats.size(); ii++) {
                if (!acc) { acc = corpus_bleu_sent_stats[ii]->GetZero(); }
                acc->PlusEquals(*corpus_bleu_sent_stats[ii]);
                
            }
            corpus_bleu_stats = acc;
            corpus_bleu_score = acc->ComputeScore();
        }
        //corpus_src_length = 0;
    }
    const DocScorer& ds;
    vector<ScoreP>& corpus_bleu_sent_stats;
    vector<GoodBadOracle>& oracles;
    vector<shared_ptr<HypothesisInfo> > cur_best;
    shared_ptr<HypothesisInfo> cur_oracle;
    const int kbest_size;
    Hypergraph forest;
    int cur_sent;
    ScoreP corpus_bleu_stats;
    float corpus_bleu_score;
    
    //float corpus_src_length;
    //float curr_src_length;
    
    const int GetCurrentSent() const {
        return cur_sent;
    }
    
    const HypothesisInfo& GetCurrentBestHypothesis() const {
        return *cur_best[0];
    }
    
    const vector<shared_ptr<HypothesisInfo> > GetCurrentBest() const {
        return cur_best;
    }
    
    const HypothesisInfo& GetCurrentOracle() const {
        return *cur_oracle;
    }
    
    const Hypergraph& GetCurrentForest() const {
        return forest;
    }
    
    
    virtual void NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg) {
        cur_sent = smeta.GetSentenceID();
        //cerr << "SOURCE " << smeta.GetSourceLength() << endl;
        //curr_src_length = (float) smeta.GetSourceLength();
        //UpdateOracles(smeta.GetSentenceID(), *hg);
        if(unique_kbest)
            UpdateOracles<KBest::FilterUnique>(smeta.GetSentenceID(), *hg);
        else
            UpdateOracles<KBest::NoFilter<std::vector<WordID> > >(smeta.GetSentenceID(), *hg);
        forest = *hg;
        
    }
    
    shared_ptr<HypothesisInfo> MakeHypothesisInfo(const SparseVector<double>& feats, const double score, const vector<WordID>& hyp) {
        shared_ptr<HypothesisInfo> h(new HypothesisInfo);
        h->features = feats;
        h->mt_metric = score;
        h->hyp = hyp;
        return h;
    }
    
    template <class Filter>  
    void UpdateOracles(int sent_id, const Hypergraph& forest) {
        
        vector<shared_ptr<HypothesisInfo> >& cur_good = oracles[sent_id].good;
        vector<shared_ptr<HypothesisInfo> >& cur_bad = oracles[sent_id].bad;
        //TODO: look at keeping previous iterations hypothesis lists around
        cur_best.clear();
        cur_good.clear();
        cur_bad.clear();
        
        vector<shared_ptr<HypothesisInfo> > all_hyp;
        
        typedef KBest::KBestDerivations<vector<WordID>, ESentenceTraversal,Filter> K;
        K kbest(forest,kbest_size);
        
        //KBest::KBestDerivations<vector<WordID>, ESentenceTraversal> kbest(forest, kbest_size);
        for (int i = 0; i < kbest_size; ++i) {
            //const KBest::KBestDerivations<vector<WordID>, ESentenceTraversal>::Derivation* d =
            typename K::Derivation *d =
            kbest.LazyKthBest(forest.nodes_.size() - 1, i);
            if (!d) break;
            
            float sentscore;
            if(approx_score)
            {
                
                if(cur_pass > 0)
                {
                    ScoreP sent_stats = ds[sent_id]->ScoreCandidate(d->yield);
                    ScoreP corpus_no_best = corpus_bleu_stats->GetZero();
                    
                    corpus_bleu_stats->Subtract(*corpus_bleu_sent_stats[sent_id], &*corpus_no_best);
                    sent_stats->PlusEquals(*corpus_no_best, 0.5);
                    
                    //compute gain from new sentence in 1-best corpus
                    sentscore = mt_metric_scale * (sent_stats->ComputeScore() - corpus_bleu_score);
                }
                
                else
                {
                    //cerr << "Using sentence-level approximation - PASS - " << boost::lexical_cast<std::string>(cur_pass) << endl;
                    //approx style of computation, used for 0th iteration
                    sentscore = mt_metric_scale * (ds[sent_id]->ScoreCandidate(d->yield)->ComputeSentScore());
                }
                
                //cerr << "CORP:" << corpus_bleu_score << " NEW:" << sent_stats->ComputeScore() << " sentscore:" << sentscore << endl;
                
                //-----pseudo-corpus approach
                /*	  float src_scale = corpus_src_length + curr_src_length;
                 ScoreP sent_stats = ds[sent_id]->ScoreCandidate(d->yield);
                 if(!corpus_bleu_stats){ corpus_bleu_stats = sent_stats->GetZero();}
                 sent_stats->PlusEquals(*corpus_bleu_stats);
                 sentscore = src_scale * sent_stats->ComputeScore();*/
            }
            else
            {
                sentscore = mt_metric_scale * (ds[sent_id]->ScoreCandidate(d->yield)->ComputeScore());
            }
            
            if (invert_score) sentscore *= -1.0;
            //cerr << TD::GetString(d->yield) << " ||| " << d->score << " ||| " << sentscore << " " << approx_sentscore << endl;
            
            if (i < update_list_size){ 
                if (i == 0) //take cur best and add its bleu statistics counts to the pseudo-doc
                {  }
                cerr << TD::GetString(d->yield) << " ||| " << d->score << " ||| " << sentscore << endl; 
                cur_best.push_back( MakeHypothesisInfo(d->feature_values, sentscore, d->yield));
            }
            
            all_hyp.push_back(MakeHypothesisInfo(d->feature_values, sentscore,d->yield));   //store all hyp to extract oracle best and worst
            
        }
        
        //update psuedo-doc stats
        /*ScoreP sent_stats = ds[sent_id]->ScoreCandidate(cur_best[0]->hyp);
         corpus_bleu_stats->PlusEquals(*sent_stats);
         
         sent_stats = corpus_bleu_stats;
         corpus_bleu_stats = sent_stats->GetZero();
         corpus_bleu_stats->PlusEquals(*sent_stats, PSEUDO_SCALE);
         
         corpus_src_length = PSEUDO_SCALE * (corpus_src_length + curr_src_length);
         */
        
        bool PRINT_LIST= false;
        
        //figure out how many hyps we can keep maximum
        int temp_update_size = update_list_size;
        if (all_hyp.size() < update_list_size){ temp_update_size = all_hyp.size();}
        
        //sort all hyps by sentscore (bleu)
        sort(all_hyp.begin(),all_hyp.end(),HypothesisCompareB);
        
        if(PRINT_LIST){  cerr << "Sorting " << endl; for(int u=0;u!=all_hyp.size();u++)	cerr << all_hyp[u]->mt_metric << " " << all_hyp[u]->features << endl; }
        
        if(optimizer != 4)
        {
            //find hope hypothesis
            if (PRINT_LIST) cerr << "HOPE " << endl;
            for(int u=0;u!=all_hyp.size();u++)	
            { 
                double t_score = all_hyp[u]->features.dot(dense_weights);
                all_hyp[u]->hope = 1 * all_hyp[u]->mt_metric + t_score;
                if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " S:" << t_score << endl; 
                
            }
            
            //sort hyps by hope score
            sort(all_hyp.begin(),all_hyp.end(),HopeCompareB);
        }
        
        //assign cur_good the sorted list
        cur_good.insert(cur_good.begin(), all_hyp.begin(), all_hyp.begin()+temp_update_size);    
        if(PRINT_LIST) { cerr << "GOOD" << endl;  for(int u=0;u!=cur_good.size();u++) cerr << cur_good[u]->mt_metric << " " << cur_good[u]->hope << endl;}     
        /*    if (!cur_oracle) {      cur_oracle = cur_good[0];
         cerr << "Set oracle " << cur_oracle->hope << " " << cur_oracle->fear << " " << cur_oracle->mt_metric << endl;      }
         else      {
         cerr << "Stay oracle " << cur_oracle->hope << " " << cur_oracle->fear << " " << cur_oracle->mt_metric << endl;      }    */
        if(optimizer != 4){
            //compute fear hyps
            if (PRINT_LIST) cerr << "FEAR " << endl;
            double hope_score = all_hyp[0]->features.dot(dense_weights);
            //double hope_score = cur_oracle->features.dot(dense_weights);
            if (PRINT_LIST) cerr << "hope score " << hope_score << endl;
            for(int u=0;u!=all_hyp.size();u++)	
            { 
                double t_score = all_hyp[u]->features.dot(dense_weights);
                //all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - hope_score + t_score;
                
                /*	  all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - -1*cur_oracle->mt_metric - hope_score + t_score; //relative loss
                 all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric - -1*cur_oracle->mt_metric;
                 all_hyp[u]->oracle_feat_diff = cur_oracle->features - all_hyp[u]->features;*/
                
                all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - -1*all_hyp[0]->mt_metric - hope_score + t_score; //relative loss
                all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric - -1*all_hyp[0]->mt_metric;
                all_hyp[u]->oracle_feat_diff = all_hyp[0]->features - all_hyp[u]->features;
                //	all_hyp[u]->fear = -1 * all_hyp[u]->mt_metric + t_score;
                if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " F:" << all_hyp[u]->fear << endl; 
                
            }
            
            sort(all_hyp.begin(),all_hyp.end(),FearCompareB);
            
            cur_bad.insert(cur_bad.begin(), all_hyp.begin(), all_hyp.begin()+temp_update_size);    
        }
        else{
            cur_bad.insert(cur_bad.begin(), all_hyp.end()-temp_update_size, all_hyp.end()); 
            reverse(cur_bad.begin(),cur_bad.end());
        }
        
        
        if(PRINT_LIST){ cerr<< "BAD"<<endl; for(int u=0;u!=cur_bad.size();u++) cerr << cur_bad[u]->mt_metric << " H:" << cur_bad[u]->hope << " F:" << cur_bad[u]->fear << endl;}
        
        cerr << "GOOD (BEST): " << cur_good[0]->mt_metric << endl;
        cerr << " CUR: " << cur_best[0]->mt_metric << endl;
        cerr << " BAD (WORST): " << cur_bad[0]->mt_metric << endl;
    }
};

void ReadTrainingCorpus(const string& fname, vector<string>* c) {
    
    
    ReadFile rf(fname);
    istream& in = *rf.stream();
    string line;
    while(in) {
        getline(in, line);
        if (!in) break;
        c->push_back(line);
    }
}

void ReadPastTranslationForScore(const int cur_pass, vector<ScoreP>* c, DocScorer& ds, const string& od)
{
    cerr << "Reading BLEU gain file ";
    string fname;
    if(cur_pass == 0)
    {
        fname = od + "/run.raw.init";
    }
    else
    {
        int last_pass = cur_pass - 1; 
        fname = od + "/run.raw."  +  boost::lexical_cast<std::string>(last_pass) + ".B";
    }
    cerr << fname << "\n";
    ReadFile rf(fname);
    istream& in = *rf.stream();
    ScoreP acc;
    string line;
    int lc = 0;
    while(in) {
        getline(in, line);
        if (line.empty() && !in) break;
        vector<WordID> sent;
        TD::ConvertSentence(line, &sent);
        ScoreP sentscore = ds[lc]->ScoreCandidate(sent);
        c->push_back(sentscore);
        if (!acc) { acc = sentscore->GetZero(); }
        acc->PlusEquals(*sentscore);
        ++lc;
        
    }
    
    
    assert(lc > 0);
    float score = acc->ComputeScore();
    string details;
    acc->ScoreDetails(&details);
    cerr << "INIT RUN " << details << score << endl;
    
}


int main(int argc, char** argv) {
    register_feature_functions();
    SetSilent(true);  // turn off verbose decoder output
    
    po::variables_map conf;
    if (!InitCommandLine(argc, argv, &conf)) return 1;
    
    if (conf.count("random_seed"))
        rng.reset(new MT19937(conf["random_seed"].as<uint32_t>()));
    else
        rng.reset(new MT19937);
    
    vector<string> corpus;
    //ReadTrainingCorpus(conf["source"].as<string>(), &corpus);
    
    const string metric_name = conf["mt_metric"].as<string>();
    optimizer = conf["optimizer"].as<int>();
    mt_metric_scale = conf["mt_metric_scale"].as<double>();
    approx_score = conf.count("approx_score");
    no_reweight = conf.count("no_reweight");
    no_select = conf.count("no_select");
    update_list_size = conf["update_k_best"].as<int>();
    unique_kbest = conf.count("unique_k_best");
    
    const string weights_dir = conf["weights_output"].as<string>();
    const string output_dir = conf["output_dir"].as<string>();
    ScoreType type = ScoreTypeFromString(metric_name);
    
    if (type == TER) {
        invert_score = true;
        approx_score = false;
    } else {
        invert_score = false;
    }
    DocScorer ds(type, conf["reference"].as<vector<string> >(), "");
    cerr << "Loaded " << ds.size() << " references for scoring with " << metric_name << endl;
    vector<ScoreP> corpus_bleu_sent_stats;
    cur_pass = conf["passes"].as<int>();
    if(cur_pass > 0)
    {
        ReadPastTranslationForScore(cur_pass, &corpus_bleu_sent_stats, ds, output_dir);
    }
    /*  if (ds.size() != corpus.size()) {
     cerr << "Mismatched number of references (" << ds.size() << ") and sources (" << corpus.size() << ")\n";
     return 1;
     }*/
    cerr << "Optimizing with " << optimizer << endl;
    // load initial weights
    Weights weights;
    weights.InitFromFile(conf["input_weights"].as<string>());
    SparseVector<double> lambdas;
    weights.InitSparseVector(&lambdas);
    
    ReadFile ini_rf(conf["decoder_config"].as<string>());
    Decoder decoder(ini_rf.stream());
    
    const string input = decoder.GetConf()["input"].as<string>();
    //const bool show_feature_dictionary = decoder.GetConf().count("show_feature_dictionary");
    if (!SILENT) cerr << "Reading input from " << ((input == "-") ? "STDIN" : input.c_str()) << endl;
    ReadFile in_read(input);
    istream *in = in_read.stream();
    assert(*in);  
    string buf;
    
    //@author ferhanture: get a handle to the Discourse ff object
	vector<boost::shared_ptr<FeatureFunction> > ffs = decoder.GetFFs();
	boost::shared_ptr<Discourse> ff_discourse;
	int discourse_cnt = 0;
	
	for(int i=0; i<ffs.size(); i++){
		cerr << i << endl;
		if(ffs[i]->name_ == "Discourse"){
			cerr << "disc"<<i << endl;
			ff_discourse = boost::dynamic_pointer_cast<Discourse, FeatureFunction>(ffs[i]);
			discourse_cnt=1;
		}  
	}
    
    //@author ferhanture: run cdec w/ discourse feature
    string rulefreq_dir, rulefreq_file;
    //read num of docs
    vector<string> df_options = decoder.GetConf()["df"].as<vector<string> >();	
    string df_file = df_options[0];
    int num_docs = atoi(df_options[1].c_str());
    ff_discourse->set_df_numdocs(num_docs); 
    
    //read df values
    ReadFile in_read2(df_file);
    istream *in2 = in_read2.stream();
    string line="";
    std::map<string,int> dfs;
    while(*in2){
        getline(*in2,line);
        if (line.empty()) continue;
        vector<string> term_df;
        boost::split(term_df, line, boost::is_any_of(" "));
        string term = term_df[0];
        int df = atoi(term_df[1].c_str());
        dfs[term]=df;
    }
    ff_discourse->set_dfs(dfs); 
    
    //rule freq folder
    bool rulefreq_perdoc = decoder.GetConf().count("rules_dir");
    bool rulefreq_percollection = decoder.GetConf().count("rules_file");
    if(rulefreq_perdoc){
        rulefreq_dir = decoder.GetConf()["rules_dir"].as<string>();
    }else if(rulefreq_percollection){
        rulefreq_file = decoder.GetConf()["rules_file"].as<string>();
        ff_discourse->load_freqs(rulefreq_file);
    }
    
    const double max_step_size = conf["max_step_size"].as<double>();
    
    
    //  assert(corpus.size() > 0);
    vector<GoodBadOracle> oracles(ds.size());
    
    TrainingObserver observer(conf["k_best_size"].as<int>(), ds, &oracles, &corpus_bleu_sent_stats);
    
    int cur_sent = 0;
    int lcount = 0;
    double objective=0;
    double tot_loss = 0;
    int dots = 0;
    //  int cur_pass = 1;
    //  vector<double> dense_weights;
    SparseVector<double> tot;
    SparseVector<double> final_tot;
    //  tot += lambdas;          // initial weights
    //  lcount++;                // count for initial weights
    
    //string msg = "# MIRA tuned weights";
    // while (cur_pass <= max_iteration) {
    SparseVector<double> old_lambdas = lambdas;
    tot.clear();
    tot += lambdas;
    cerr << "PASS " << cur_pass << " " << endl << lambdas << endl; 
    ScoreP acc, acc_h, acc_f;
    
    while(*in) {
        getline(*in, buf);
        if (buf.empty()) continue;
        //for (cur_sent = 0; cur_sent < corpus.size(); cur_sent++) {
        
        //        cerr << "SENT: " << cur_sent << endl;
        
        //@author FERHANTURE
        vector<string> segments_in_doc = split(buf, "<NEXTSEG>", false);
        int num_segments = segments_in_doc.size();
        
        if(rulefreq_perdoc){
            int doc_id = get_doc_id(segments_in_doc[0]);
            ff_discourse->load_freqs(doc_id, rulefreq_dir);
        }
        
        stringstream outstrm;
        for(int i=0;i<num_segments;i++){
            string next_segment = segments_in_doc[i];
            
            //TODO: allow batch updating
            dense_weights.clear();
            weights.InitFromVector(lambdas);
            weights.InitVector(&dense_weights);
            decoder.SetWeights(dense_weights);  
            decoder.SetId(cur_sent);
            decoder.Decode(next_segment, &observer);  // update oracles
            cur_sent = observer.GetCurrentSent();
            const HypothesisInfo& cur_hyp = observer.GetCurrentBestHypothesis();
            const HypothesisInfo& cur_good = *oracles[cur_sent].good[0];
            const HypothesisInfo& cur_bad = *oracles[cur_sent].bad[0];
            
            vector<shared_ptr<HypothesisInfo> >& cur_good_v = oracles[cur_sent].good;
            vector<shared_ptr<HypothesisInfo> >& cur_bad_v = oracles[cur_sent].bad;
            vector<shared_ptr<HypothesisInfo> > cur_best_v = observer.GetCurrentBest();
            
            tot_loss += cur_hyp.mt_metric;
            
            //score cur_hyp to get corpus level bleu after pass
            ScoreP sentscore = ds[cur_sent]->ScoreCandidate(cur_hyp.hyp);
            if (!acc) { acc = sentscore->GetZero(); }
            acc->PlusEquals(*sentscore);
            
            ScoreP hope_sentscore = ds[cur_sent]->ScoreCandidate(cur_good.hyp);
            if (!acc_h) { acc_h = hope_sentscore->GetZero(); }
            acc_h->PlusEquals(*hope_sentscore);
            ScoreP fear_sentscore = ds[cur_sent]->ScoreCandidate(cur_bad.hyp);
            if (!acc_f) { acc_f = fear_sentscore->GetZero(); }
            acc_f->PlusEquals(*fear_sentscore);
            
            if(optimizer == 4) { //local update
                if (!ApproxEqual(cur_hyp.mt_metric, cur_good.mt_metric)) {
                    
                    double margin = cur_bad.features.dot(dense_weights) - cur_good.features.dot(dense_weights);
                    double mt_loss = mt_metric_scale * (cur_good.mt_metric - cur_bad.mt_metric);
                    const double loss = margin +  mt_loss;
                    cerr << "LOSS: " << loss << " Margin:" << margin << " BLEUL:" << mt_loss << " " << cur_bad.features.dot(dense_weights) << " " << cur_good.features.dot(dense_weights) <<endl;
                    if (loss > 0.0) {
                        SparseVector<double> diff = cur_good.features;
                        diff -= cur_bad.features;
                        double step_size = loss / diff.l2norm_sq();
                        cerr << loss << " " << step_size << " " << diff << endl;
                        if (step_size > max_step_size) step_size = max_step_size;
                        lambdas += (cur_good.features * step_size);
                        lambdas -= (cur_bad.features * step_size);
                        //cerr << "L: " << lambdas << endl;
                    }
                }
            }
            else if(optimizer == 1) //sgd
            {
                
                lambdas += (cur_good.features) * max_step_size;
                lambdas -= (cur_bad.features) * max_step_size;
            }
            //cerr << "L: " << lambdas << endl;
            else if(optimizer == 5) //full mira with n-best list of constraints from oracle, fear, best
            {
                vector<shared_ptr<HypothesisInfo> > cur_constraint;
                cur_constraint.insert(cur_constraint.begin(), cur_bad_v.begin(), cur_bad_v.end());
                cur_constraint.insert(cur_constraint.begin(), cur_best_v.begin(), cur_best_v.end());
                cur_constraint.insert(cur_constraint.begin(), cur_good_v.begin(), cur_good_v.end());
                
                bool optimize_again;
                vector<shared_ptr<HypothesisInfo> > cur_pair;
                //SMO 
                for(int u=0;u!=cur_constraint.size();u++)	
                    cur_constraint[u]->alpha =0;	      
                
                cur_constraint[0]->alpha =1; //set oracle to alpha=1
                
                cerr <<"Optimizing with " << cur_constraint.size() << " constraints" << endl;
                int smo_iter = 10, smo_iter2 = 10;
                int iter, iter2 =0;
                bool DEBUG_SMO = false;
                while (iter2 < smo_iter2)
                {
                    iter =0;
                    while (iter < smo_iter)
                    {
                        optimize_again = true;
                        for (int i = 0; i< cur_constraint.size(); i++)
                            for (int j = i+1; j< cur_constraint.size(); j++)
                            {
                                if(DEBUG_SMO) cerr << "start " << i << " " << j <<  endl;
                                cur_pair.clear();
                                cur_pair.push_back(cur_constraint[j]);
                                cur_pair.push_back(cur_constraint[i]);
                                double delta = ComputeDelta(&cur_pair,max_step_size);
                                
                                if (delta == 0) optimize_again = false;
                                //			cur_pair[0]->alpha += delta;
                                //	cur_pair[1]->alpha -= delta;
                                cur_constraint[j]->alpha += delta;
                                cur_constraint[i]->alpha -= delta;
                                double step_size = delta * max_step_size;
                                /*lambdas += (cur_pair[1]->features) * step_size;
                                 lambdas -= (cur_pair[0]->features) * step_size;*/
                                lambdas += (cur_constraint[i]->features) * step_size;
                                lambdas -= (cur_constraint[j]->features) * step_size;
                                if(DEBUG_SMO) cerr << "SMO opt " << iter << " " << i << " " << j << " " <<  delta << " " << cur_pair[0]->alpha << " " << cur_pair[1]->alpha <<  endl;		
                                
                                //reload weights based on update
                                dense_weights.clear();
                                weights.InitFromVector(lambdas);
                                weights.InitVector(&dense_weights);
                            }
                        iter++;
                        
                        if(!optimize_again)
                        { 
                            iter = 100;
                            cerr << "Optimization stopped, delta =0" << endl;
                        }
                        
                        
                    }
                    iter2++;
                }
                
                
            }
            else if(optimizer == 2 || optimizer == 3) //1-fear and cutting plane mira
            {
                bool DEBUG_SMO= true;
                vector<shared_ptr<HypothesisInfo> > cur_constraint;
                cur_constraint.push_back(cur_good_v[0]); //add oracle to constraint set
                bool optimize_again = true;
                while (optimize_again)
                { 
                    if(DEBUG_SMO) cerr<< "optimize again: " << optimize_again << endl;
                    if(optimizer == 2){ //1-fear
                        cur_constraint.push_back(cur_bad_v[0]);
                    }
                    else
                    { //cutting plane to add constraints
                        if(DEBUG_SMO) cerr<< "Cutting Plane with " << lambdas << endl;
                        optimize_again = false;
                        CuttingPlane(&cur_constraint, &optimize_again, oracles[cur_sent].bad, dense_weights);
                    }
                    
                    if(optimize_again)
                    {
                        //SMO 
                        for(int u=0;u!=cur_constraint.size();u++)	
                        { 
                            cur_constraint[u]->alpha =0;
                            //cur_good_v[0]->alpha = 1; cur_bad_v[0]->alpha = 0;
                        }
                        cur_constraint[0]->alpha = 1;
                        cerr <<"Optimizing with " << cur_constraint.size() << " constraints" << endl;
                        int smo_iter = MAX_SMO;
                        int iter =0;
                        while (iter < smo_iter)
                        {
                            
                            //select pair to optimize from constraint set
                            vector<shared_ptr<HypothesisInfo> > cur_pair = SelectPair(&cur_constraint);
                            
                            if(cur_pair.empty()){iter=MAX_SMO; cerr << "Undefined pair " << endl; continue;} //pair is undefined so we are done with this smo 
                            
                            //double num = cur_good_v[0]->fear - cur_bad_v[0]->fear;
                            /*double loss = cur_good_v[0]->oracle_loss - cur_bad_v[0]->oracle_loss;
                             double margin = cur_good_v[0]->oracle_feat_diff.dot(dense_weights) - cur_bad_v[0]->oracle_feat_diff.dot(dense_weights);
                             double num = loss - margin;
                             SparseVector<double> diff = cur_good_v[0]->features;
                             diff -= cur_bad_v[0]->features;
                             double delta = num / (diff.l2norm_sq() * max_step_size);
                             delta = max(-cur_good_v[0]->alpha, min(delta, cur_bad_v[0]->alpha));
                             cur_good_v[0]->alpha += delta;
                             cur_bad_v[0]->alpha -= delta;
                             double step_size = delta * max_step_size;
                             lambdas += (cur_bad_v[0]->features) * step_size;
                             lambdas -= (cur_good_v[0]->features) * step_size;
                             */
                            
                            double delta = ComputeDelta(&cur_pair,max_step_size);
                            
                            cur_pair[0]->alpha += delta;
                            cur_pair[1]->alpha -= delta;
                            double step_size = delta * max_step_size;
                            /*			lambdas += (cur_pair[1]->oracle_feat_diff) * step_size;
                             lambdas -= (cur_pair[0]->oracle_feat_diff) * step_size;*/
                            
                            double alpha_sum=0;
                            SparseVector<double> temp_lambdas = lambdas;
                            
                            for(int u=0;u!=cur_constraint.size();u++)	
                            { 
                                cerr << cur_constraint[u]->alpha << " " << cur_constraint[u]->hope << endl;
                                temp_lambdas += (cur_constraint[u]->oracle_feat_diff) * cur_constraint[u]->alpha * max_step_size;
                                alpha_sum += cur_constraint[u]->alpha;
                            }
                            cerr << "Alpha sum " << alpha_sum << " " << temp_lambdas << endl;
                            
                            
                            lambdas += (cur_pair[1]->features) * step_size;
                            lambdas -= (cur_pair[0]->features) * step_size;
                            cerr << " Lambdas " << lambdas << endl;
                            
                            SparseVector<double> diff = cur_pair[0]->features;
                            diff -= cur_pair[1]->features;
                            cerr << "0:" << cur_pair[0]->features << endl;
                            cerr << "1:" << cur_pair[1]->features << endl;
                            cerr << "diff:" << diff << endl;
                            
                            
                            //reload weights based on update
                            dense_weights.clear();
                            weights.InitFromVector(lambdas);
                            weights.InitVector(&dense_weights);
                            iter++;
                            
                            
                            
                            
                            if(DEBUG_SMO) cerr << "SMO opt " << iter << " " << delta << " " << cur_pair[0]->alpha << " " << cur_pair[1]->alpha <<  endl;		
                            //		cerr << "SMO opt " << iter << " " << delta << " " << cur_good_v[0]->alpha << " " << cur_bad_v[0]->alpha <<  endl;
                            if(no_select) //don't use selection heuristic to determine when to stop SMO, rather just when delta =0 
                                if (delta == 0) iter = MAX_SMO;
                            
                        }
                        if(optimizer == 3)
                        {
                            if(!no_reweight)
                            {
                                if(DEBUG_SMO) cerr<< "Decoding with new weights -- now orac are " << oracles[cur_sent].good.size() << endl;
                                Hypergraph hg = observer.GetCurrentForest();
                                hg.Reweight(dense_weights);
                                //observer.UpdateOracles(cur_sent, hg);
                                if(unique_kbest)
                                    observer.UpdateOracles<KBest::FilterUnique>(cur_sent, hg);
                                else
                                    observer.UpdateOracles<KBest::NoFilter<std::vector<WordID> > >(cur_sent, hg);
                                
                                
                            }
                        }
                    }
                    
                    if(optimizer == 2) optimize_again = false;
                    
                }
                
                //print objective after this sentence
                double lambda_change = (lambdas - old_lambdas).l2norm_sq();
                double max_fear = cur_constraint[cur_constraint.size()-1]->fear;
                double temp_objective = 0.5 * lambda_change;// + max_step_size * max_fear;
                
                for(int u=0;u!=cur_constraint.size();u++)	
                { 
                    cerr << cur_constraint[u]->alpha << " " << cur_constraint[u]->hope << " " << cur_constraint[u]->fear << endl;
                    temp_objective += cur_constraint[u]->alpha * cur_constraint[u]->fear;
                }
                objective += temp_objective;
                
                cerr << "SENT OBJ: " << temp_objective << " NEW OBJ: " << objective << endl;
            }
            
            if ((cur_sent * 40 / ds.size()) > dots) { ++dots; cerr << '.'; }
            tot += lambdas;
            ++lcount;
            cur_sent++;
            
            if(i==num_segments-1){
                outstrm << TD::GetString(cur_good_v[0]->hyp) << " ||| " << TD::GetString(cur_best_v[0]->hyp) << " ||| " << TD::GetString(cur_bad_v[0]->hyp);
            }else{
                outstrm << TD::GetString(cur_good_v[0]->hyp) << " ||| " << TD::GetString(cur_best_v[0]->hyp) << " ||| " << TD::GetString(cur_bad_v[0]->hyp) << "<NEXTSEG>";
            }
        }
        cerr << "doc translation done." << endl;
        cout << outstrm.str() << endl;
        
        //reset the list of segments and translations in memory
        segments_in_doc.clear();
        decoder.NewDocument();
    }
    //END @author ferhanture
    
    final_tot += tot;
    cerr << "Translated " << lcount << " sentences " << endl;
    cerr << " [AVG METRIC LAST PASS=" << (tot_loss / lcount) << "]\n";
    tot_loss = 0;
    /*
     float corpus_score = acc->ComputeScore();
     string corpus_details;
     acc->ScoreDetails(&corpus_details);
     cerr << "MODEL " << corpus_details << endl;
     cout << corpus_score << endl;
     
     corpus_score = acc_h->ComputeScore();
     acc_h->ScoreDetails(&corpus_details);
     cerr << "HOPE " << corpus_details << endl;
     cout << corpus_score << endl;
     
     corpus_score = acc_f->ComputeScore();
     acc_f->ScoreDetails(&corpus_details);
     cerr << "FEAR " << corpus_details << endl;
     cout << corpus_score << endl;
     */
    int node_id = rng->next() * 100000;
    cerr << " Writing weights to " << node_id << endl;
    dots = 0;
    ostringstream os;
    os << weights_dir << "/weights.mira-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << "." << node_id << ".gz";
    string msg = "# MIRA tuned weights ||| " + boost::lexical_cast<std::string>(node_id) + " ||| " + boost::lexical_cast<std::string>(lcount);
    
    weights.WriteToFile(os.str(), true, &msg);
    
    SparseVector<double> x = tot;
    x /= cur_sent+1;
    ostringstream sa;
    string msga = "# MIRA tuned weights AVERAGED";
    sa << weights_dir << "/weights.mira-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << "." << node_id << "-avg.gz";
    Weights ww;
    ww.InitFromVector(x);
    ww.WriteToFile(sa.str(), true, &msga);
    
    //assign averaged lambdas to initialize next iteration
    //lambdas = x;
    
    /*    double lambda_change = (old_lambdas - lambdas).l2norm_sq();
     cerr << "Change in lambda " << lambda_change << endl;
     
     if ( lambda_change < EPSILON)
     {
     cur_pass = max_iteration;
     cerr << "Weights converged - breaking" << endl;
     }
     
     ++cur_pass;
     */
    
    //} iteration while loop
    
    /* cerr << endl;
     weights.WriteToFile("weights.mira-final.gz", true, &msg);
     final_tot /= (lcount + 1);//max_iteration);
     tot /= (corpus.size() + 1);
     weights.InitFromVector(final_tot);
     cerr << tot << "||||" << final_tot << endl;
     msg = "# MIRA tuned weights (averaged vector)";
     weights.WriteToFile("weights.mira-final-avg.gz", true, &msg);
     */
    cerr << "Optimization complete.\\AVERAGED WEIGHTS: weights.mira-final-avg.gz\n";
    return 0;
}

int get_doc_id(string s){  
	// get segment id from tag
	map<string, string> sgml;
	int doc_id;
	ProcessAndStripSGML(&s, &sgml);
	if (sgml.find("docid") != sgml.end()){
		doc_id = atoi(sgml["docid"].c_str());
	}else{
		cerr << "Discourse feature only works with docid tags. Include docid tags in segments and re-run";
		exit(EXIT_FAILURE);
	}
	return doc_id;
}

int get_sent_id(string s){  
	// get segment id from tag
	map<string, string> sgml;
	int sent_id;
	ProcessAndStripSGML(&s, &sgml);
	if (sgml.find("id") != sgml.end()){
		sent_id = atoi(sgml["id"].c_str());
	}else{
		cerr << "Discourse feature only works with id tags. Include id tags in segments and re-run";
		remove("rules.out");
		exit(EXIT_FAILURE);
	}
	return sent_id;
}


//code taken from http://stackoverflow.com/questions/236129/how-to-split-a-string
//split string by a multicharacter delimiter (not possible with builtin split fns)
vector<string> split(const string& s, const string& delim, const bool keep_empty) {
	vector<string> result;
	if (delim.empty()) {
		result.push_back(s);
		return result;
	}
	string::const_iterator substart = s.begin(), subend;
	while (true) {
		subend = search(substart, s.end(), delim.begin(), delim.end());
		string temp(substart, subend);
		if (keep_empty || !temp.empty()) {
			result.push_back(temp);
		}
		if (subend == s.end()) {
			break;
		}
		substart = subend + delim.size();
	}
	return result;
}

