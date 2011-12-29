//TODO: non-sparse vector for all feature functions?  modelset applymodels keeps track of who has what features?  it's nice having FF that could generate a handful out of 10000 possible feats, though.

//TODO: actually score rule_feature()==true features once only, hash keyed on rule or modify TRule directly?  need to keep clear in forest which features come from models vs. rules; then rescoring could drop all the old models features at once
#include "fast_lexical_cast.hpp"
#include <stdexcept>
#include "ff.h"
#include "tdict.h"
#include "hg.h"
#include <iostream>
#include <map>
#include <boost/foreach.hpp>
#include "filelib.h"
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string.hpp>
#include <sstream>

using namespace std;

typedef std::map<std::string, float> MapType;
typedef std::map<std::string, int> MapType2;

FeatureFunction::~FeatureFunction() {}

void FeatureFunction::PrepareForInput(const SentenceMetadata&) {}

void FeatureFunction::FinalTraversalFeatures(const void* /* ant_state */,
                                             SparseVector<double>* /* features */) const {
}

string FeatureFunction::usage_helper(std::string const& name,std::string const& params,std::string const& details,bool sp,bool sd) {
  string r=name;
  if (sp) {
    r+=": ";
    r+=params;
  }
  if (sd) {
    r+="\n";
    r+=details;
  }
  return r;
}

Features FeatureFunction::single_feature(WordID feat) {
  return Features(1,feat);
}

Features ModelSet::all_features(std::ostream *warn,bool warn0) {
  return ::all_features(models_,weights_,warn,warn0);
}

void show_features(Features const& ffs,DenseWeightVector const& weights_,std::ostream &out,std::ostream &warn,bool warn_zero_wt) {
  out << "Weight  Feature\n";
  for (unsigned i=0;i<ffs.size();++i) {
    WordID fid=ffs[i];
    string const& fname=FD::Convert(fid);
    double wt=weights_[fid];
    if (warn_zero_wt && wt==0)
      warn<<"WARNING: "<<fname<<" has 0 weight."<<endl;
    out << wt << "  " << fname<<endl;
  }
}


//////////////////////-------------------===================++++++++++++++++++++++++++///////////////////
//////////////////////-------------------===================++++++++++++++++++++++++++///////////////////
//////////////////////-------------------===================++++++++++++++++++++++++++///////////////////


Discourse::Discourse(const string& param) :
fid_(FD::Convert("Discourse")){
	if (!param.empty()) {
		cerr << "Warning Discourse ignoring parameter: " << param << endl;
	}
	string fname = "Discourse_";
    
    cerr << "initializing Discourse feature...\n";
    
    isDisc.push_back(true);
    isDisc.push_back(true);
    isDisc.push_back(true);
    
	//default values
	for (unsigned i = 0; i <= 2; ++i) {
        if(isDisc[i]){
            WordID fid=FD::Convert(fname+lexical_cast<string>(i));
            cerr << i << "->" << fid << endl;
            fids_.push_back(fid);
        }
	}
    
    cerr << "initialized, number of discourse variations:" << fids_.size() << endl;
    
	while (!fids_.empty() && fids_.back()==0) fids_.pop_back(); // pretty up features vector in case FD was frozen.  doesn't change anything
}

Features Discourse::features() const {
	return Features(fids_.begin(),fids_.end());
}

void Discourse::load_freqs(std::string rulesfile){
    for(int i=0; i<3; i++){
        MapType r;
        //if isDisc[i] is false, then add empty MapType object to rule_freq -- we use a fixed size with non-null objects for easy implementation
        if(isDisc[i]){

            stringstream rulefreqfile;
            rulefreqfile << rulesfile << i;
            cerr << "loading rule freqs from " << rulefreqfile.str() << "...";
            ReadFile in_read(rulefreqfile.str());
            istream *in = in_read.stream();
            string line="";
            int cnt = 0;
            while(*in){
                getline(*in,line);
                vector<string> arr;
                boost::split(arr, line, boost::is_any_of("="));
                string rule = arr[0];
                int count = atoi(arr[1].c_str());

                r[rule]=count;
                if(DEBUG)
                    cerr << rule << "-->" << count << endl;

                cnt++;
            }
            cerr << cnt << " freqs loaded." << endl;
        }
        rule_freq.push_back(r);
    }
}

void Discourse::load_freqs(int docid, std::string rulesdir){
    for(int i=0; i<3; i++){
        MapType r;
        //if isDisc[i] is false, then add empty MapType object to rule_freq -- we use a fixed size with non-null objects for easy implementation
        if(isDisc[i]){

            stringstream rulefreqfile;
            rulefreqfile << rulesdir<<i<<"/rulefreq_doc" << "." << docid;
            cerr << "loading rule freqs from " << rulefreqfile.str() << "...";
            ReadFile in_read(rulefreqfile.str());
            istream *in = in_read.stream();
            string line="";
            int cnt = 0;
            while(*in){
                getline(*in,line);
                vector<string> arr;
                boost::split(arr, line, boost::is_any_of("="));
                string rule = arr[0];
                int count = atoi(arr[1].c_str());
                
                r[rule]=count;
                if(DEBUG)
                    cerr << rule << "-->" << count << endl;
                
                cnt++;
            }
            cerr << cnt << " freqs loaded." << endl;
        }
        rule_freq.push_back(r);
    }
}

void Discourse::PrepareForInput(const SentenceMetadata& smeta) { }

//no weight in rules read from TraversalFeaturesImpl(): where we score rules
std::vector<std::string> Discourse::parse_rule_into_vector(const int mode, const std::string &line) const{
	vector<string> countables;
	
	//1. find the rhs of the rule
	size_t pos1 = line.find("|||");
	string rule1 = line.substr(pos1+3,line.length());
    size_t pos2 = rule1.find("|||");
    string rule2 = rule1.substr(pos2+3,rule1.length());
    size_t pos3 = rule2.find("|||");
    
    string rule = line.substr(pos1+4,pos2+3+pos3-2);
    
    //extract alignments from rule string
    MapType alignMap;
    if(mode == 2){
        size_t pos = line.rfind("|||");
    	string alignments = line.substr(pos+4,line.length());
    	std::vector<std::string> aligns;
    	boost::split(aligns, alignments, boost::is_any_of(" "));

    	for(int k=0; k<aligns.size();k++){
       	 alignMap[aligns[k]]=1;
    	}
    }
    
    if(DEBUG)
        cerr << "Read="<<rule << "=" << endl;
	
	if(mode==0){							// LHS ||| RHS
        if(DEBUG)
            cerr << "countable=" << rule << endl; 
        countables.push_back(rule);			
	}else{						
		size_t pos3= rule.find("|||");
		size_t pos4 = rule.rfind("|||");
		string lhs = rule.substr(0,pos3);
		string rhs = rule.substr(pos3+3,rule.length()-pos3-3);
		boost::algorithm::trim(rhs);
		boost::algorithm::trim(lhs);
		vector<string> rhs_words;
        boost::split(rhs_words, rhs, boost::is_any_of(" "));
		
        
		if (mode==1) {						// RHS_1 ; RHS_2 ; ...
			//2. split rhs into words
			
			for (int i=0; i<rhs_words.size(); i++) {
				string word = rhs_words[i];
				
				//ignore non-terminals and punctuations
				if(!(word[0]=='[' && word[word.length()-1]==']')){
                    countables.push_back(word);
                    if(DEBUG)
						cerr << "countable=" << word << endl;			
				}
			}		
		}else if (mode==2) {				// LHS_i,RHS_j ; ...
            vector<string> lhs_words;
            boost::split(lhs_words, lhs, boost::is_any_of(" "));
            
            
            for (int i=0; i<lhs_words.size(); i++) {
				string lhs_word = lhs_words[i];
				if(!((lhs_word[0]=='[' && lhs_word[lhs_word.length()-1]==']') || lhs_word=="," || lhs_word==";" || lhs_word==":")){
					for (int j=0; j<rhs_words.size(); j++) {
                        stringstream al;
                        al << i<<"-"<<j;
                        if(alignMap[al.str()]==1){
                            string rhs_word = rhs_words[j];
                            if(!(rhs_word[0]=='[' && rhs_word[rhs_word.length()-1]==']')){
                                string countable = lhs_word+","+rhs_word;
                                if(DEBUG)
                                    cerr << "countable=" << countable << endl;
                                countables.push_back(countable);
                            }
                        }
					}
				}
			}
			
		}
	}
	return countables;
}

bool Discourse::is_glue(const std::string &line) const{
	if(line.find("[S]") != string::npos || line.find("[Goal]") != string::npos){
		return true;
	}else {
		return false;
	}	
}

void Discourse::TraversalFeaturesImpl(const SentenceMetadata& smeta,
									  const Hypergraph::Edge& edge,
									  const std::vector<const void*>& ant_states,
									  SparseVector<double>* features,
									  SparseVector<double>* estimated_features,
									  void* state) const {
	(void) smeta;
	(void) ant_states;
	(void) state;
	(void) estimated_features;
	
    string rule = edge.rule_->AsString();
    
    if(is_glue(rule)){
        for (int i=0; i<=2; i++) {
            if(isDisc[i])
                features->set_value(fids_[0], 0.0);
        }
        return;
    }
    
    int cntr = 0;
    for (int i=0; i<=2; i++) {
        if(isDisc[i]){
            float fval = compute_fval(i, rule, rule_freq[i]);
            
            if(DEBUG){
                cerr << fval << "<--v" << i << endl;
            }
            
            features->set_value(fids_[cntr], fval);
            cntr++;
        }
    }
}

float Discourse::compute_fval(int mode, string rule, MapType rule_freq) const{
	vector<string> countables = parse_rule_into_vector(mode, rule);
	
	if(countables.size() == 0)	return 0.0;
	
	if(DEBUG)	
		cerr << countables.size() << " words parsed." << endl;
	
	// a rule contains 1 or more tokens. we define the score of a rule as the score of the max-scored token. 
	// otherwise, if there's a really important word in the rule, but many other common words, 
	// then the average score will be much lower, which is not desired. we simply want to ignore the common words in that case.
	float max_val = 0, avg_val = 0, sum_val;
	for (int i=0; i<countables.size(); i++) {
		string countable = countables[i];
        if(DEBUG)	
            cerr << "scoring " << countable << endl;	
		// ones with tf=0 are removed by parse_rule_into_vector()
		int tf = 0;
		MapType::const_iterator iter = rule_freq.find(countable);
		if(iter != rule_freq.end()){
			tf = iter->second;
			if(DEBUG)	
				cerr << "TF=" << tf << endl;
		}else{
			if(DEBUG)	
				cerr << "OOV" << endl;
			continue;
		}
		
		// one way to filter unnecessary words from RHS is to put a df threshold. e.g. score words with df<X or score word with least df
		// current solution is to take max of scores, but in this case rules like "X -> the Y" get a positive score if Y has a count of 0, but we would like to ignore that score coming from "the"
		int df = 0;
		if(	mode==0){						// countable = LHS ||| RHS. df is undefined.
			df = 0;
		}else{
			string rhs_word;
			if(mode==1){					// countable = RHS_word
				rhs_word = countable;
			}else if(mode==2){				// countable = LHS_word,RHS_word
                vector<string> rhs_words;
                boost::split(rhs_words, countable, boost::is_any_of(","));
                rhs_word = rhs_words[1];
			}
			MapType2::const_iterator iter2 = dfs.find(rhs_word);
			if (iter2 != dfs.end() ){ 
				df = iter2->second;
			}
		}
		float feat_val = bm25(tf,df);
		if(DEBUG)	
			cerr << "word " << i << " = " << countable << "," << tf << "," << df << "=" << feat_val << endl;
        
        sum_val += feat_val;
            
        if(feat_val > max_val){
            max_val = feat_val;
        }
        
	}	
    avg_val = sum_val/countables.size();
    if(DEBUG){
        cerr << "Max val = " << max_val << endl;
        cerr << "Avg val = " << avg_val << endl;
        cerr << "Sum val = " << sum_val << endl;
    }
    
	return max_val;
}

float Discourse::bm25(int tf) const{
	float K = 1.2;		//assume doclen = avg_doclen
	float val = ((K + 1.0f) * tf / (K + tf));
	return val+1;           //cdec may act weirdly for a zero value? I couldn't verify, but this'll make it safer
}

float Discourse::bm25(int tf, int df) const{
	float K = 1.2;		//assume doclen = avg_doclen
	// we add 1 to the numerator, so that idf never becomes 0. we still want to be able to get a positive score for "a ||| the"
	float idf = log((num_docs+1.0f) / (df + 0.5f));
	float val = ((K + 1.0f) * tf / (K + tf));
	return idf*val;
}



//////////////////////-------------------===================++++++++++++++++++++++++++///////////////////
//////////////////////-------------------===================++++++++++++++++++++++++++///////////////////
//////////////////////-------------------===================++++++++++++++++++++++++++///////////////////


void ModelSet::show_features(std::ostream &out,std::ostream &warn,bool warn_zero_wt)
{
//  ::show_features(all_features(),weights_,out,warn,warn_zero_wt);
  show_all_features(models_,weights_,out,warn,warn_zero_wt,warn_zero_wt);
}

// Hiero and Joshua use log_10(e) as the value, so I do to
WordPenalty::WordPenalty(const string& param) :
  fid_(FD::Convert("WordPenalty")),
    value_(-1.0 / log(10)) {
  if (!param.empty()) {
    cerr << "Warning WordPenalty ignoring parameter: " << param << endl;
  }
}

void FeatureFunction::TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                        const Hypergraph::Edge& edge,
                                        const std::vector<const void*>& ant_states,
                                        SparseVector<double>* features,
                                        SparseVector<double>* estimated_features,
                                        void* state) const {
  throw std::runtime_error("TraversalFeaturesImpl not implemented - override it or TraversalFeaturesLog.\n");
  abort();
}


void WordPenalty::TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                        const Hypergraph::Edge& edge,
                                        const std::vector<const void*>& ant_states,
                                        SparseVector<double>* features,
                                        SparseVector<double>* estimated_features,
                                        void* state) const {
  (void) smeta;
  (void) ant_states;
  (void) state;
  (void) estimated_features;
  features->set_value(fid_, edge.rule_->EWords() * value_);
}

SourceWordPenalty::SourceWordPenalty(const string& param) :
    fid_(FD::Convert("SourceWordPenalty")),
    value_(-1.0 / log(10)) {
  if (!param.empty()) {
    cerr << "Warning SourceWordPenalty ignoring parameter: " << param << endl;
  }
}

Features SourceWordPenalty::features() const {
  return single_feature(fid_);
}

Features WordPenalty::features() const {
  return single_feature(fid_);
}


void SourceWordPenalty::TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                        const Hypergraph::Edge& edge,
                                        const std::vector<const void*>& ant_states,
                                        SparseVector<double>* features,
                                        SparseVector<double>* estimated_features,
                                        void* state) const {
  (void) smeta;
  (void) ant_states;
  (void) state;
  (void) estimated_features;
  features->set_value(fid_, edge.rule_->FWords() * value_);
}

ArityPenalty::ArityPenalty(const std::string& param) :
    value_(-1.0 / log(10)) {
  string fname = "Arity_";
  unsigned MAX=DEFAULT_MAX_ARITY;
  using namespace boost;
  if (!param.empty())
    MAX=lexical_cast<unsigned>(param);
  for (unsigned i = 0; i <= MAX; ++i) {
    WordID fid=FD::Convert(fname+lexical_cast<string>(i));
    fids_.push_back(fid);
  }
  while (!fids_.empty() && fids_.back()==0) fids_.pop_back(); // pretty up features vector in case FD was frozen.  doesn't change anything
}

Features ArityPenalty::features() const {
  return Features(fids_.begin(),fids_.end());
}

void ArityPenalty::TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                         const Hypergraph::Edge& edge,
                                         const std::vector<const void*>& ant_states,
                                         SparseVector<double>* features,
                                         SparseVector<double>* estimated_features,
                                         void* state) const {
  (void) smeta;
  (void) ant_states;
  (void) state;
  (void) estimated_features;
  unsigned a=edge.Arity();
  features->set_value(a<fids_.size()?fids_[a]:0, value_);
}

ModelSet::ModelSet(const vector<double>& w, const vector<const FeatureFunction*>& models) :
    models_(models),
    weights_(w),
    state_size_(0),
    model_state_pos_(models.size()) {
  for (int i = 0; i < models_.size(); ++i) {
    model_state_pos_[i] = state_size_;
    state_size_ += models_[i]->NumBytesContext();
  }
}

void ModelSet::PrepareForInput(const SentenceMetadata& smeta) {
  for (int i = 0; i < models_.size(); ++i)
    const_cast<FeatureFunction*>(models_[i])->PrepareForInput(smeta);
}

void ModelSet::AddFeaturesToEdge(const SentenceMetadata& smeta,
                                 const Hypergraph& /* hg */,
                                 const FFStates& node_states,
                                 Hypergraph::Edge* edge,
                                 FFState* context,
                                 prob_t* combination_cost_estimate) const {
  edge->reset_info();
  context->resize(state_size_);
  if (state_size_ > 0) {
    memset(&(*context)[0], 0, state_size_);
  }
  SparseVector<double> est_vals;  // only computed if combination_cost_estimate is non-NULL
  if (combination_cost_estimate) *combination_cost_estimate = prob_t::One();
  for (int i = 0; i < models_.size(); ++i) {
    const FeatureFunction& ff = *models_[i];
    void* cur_ff_context = NULL;
    vector<const void*> ants(edge->tail_nodes_.size());
    bool has_context = ff.NumBytesContext() > 0;
    if (has_context) {
      int spos = model_state_pos_[i];
      cur_ff_context = &(*context)[spos];
      for (int i = 0; i < ants.size(); ++i) {
        ants[i] = &node_states[edge->tail_nodes_[i]][spos];
      }
    }
    ff.TraversalFeatures(smeta, *edge, ants, &edge->feature_values_, &est_vals, cur_ff_context);
  }
  if (combination_cost_estimate)
    combination_cost_estimate->logeq(est_vals.dot(weights_));
  edge->edge_prob_.logeq(edge->feature_values_.dot(weights_));
}

void ModelSet::AddFinalFeatures(const FFState& state, Hypergraph::Edge* edge,SentenceMetadata const& smeta) const {
  assert(1 == edge->rule_->Arity());
  edge->reset_info();
  for (int i = 0; i < models_.size(); ++i) {
    const FeatureFunction& ff = *models_[i];
    const void* ant_state = NULL;
    bool has_context = ff.NumBytesContext() > 0;
    if (has_context) {
      int spos = model_state_pos_[i];
      ant_state = &state[spos];
    }
    ff.FinalTraversalFeatures(smeta, *edge, ant_state, &edge->feature_values_);
  }
  edge->edge_prob_.logeq(edge->feature_values_.dot(weights_));
}

