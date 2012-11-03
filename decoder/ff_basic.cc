#include "ff_basic.h"

#include "fast_lexical_cast.hpp"
#include "hg.h"
#include "fast_lexical_cast.hpp"
#include <stdexcept>
#include "ff.h"

#include <iostream>
#include <map>
#include <boost/foreach.hpp>
#include "filelib.h"
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string.hpp>
#include <sstream>

#include "tdict.h"

using namespace std;

// @author ferhanture
typedef std::map<std::string, float> MapType;
typedef std::map<std::string, int> MapType2;

// Hiero and Joshua use log_10(e) as the value, so I do to
WordPenalty::WordPenalty(const string& param) :
  fid_(FD::Convert("WordPenalty")),
    value_(-1.0 / log(10)) {
  if (!param.empty()) {
    cerr << "Warning WordPenalty ignoring parameter: " << param << endl;
  }
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

//////////////////////===============++++++++++++++++++++++++++///////////////////


Discourse::Discourse(const string& param) :
fid_(FD::Convert("Discourse")){
	string fname = "Discourse_";
    
    cerr << "initializing Discourse feature with parameters: " << param << endl;
            
    vector<string> arr;
    boost::split(arr, param, boost::is_any_of(" "));
    cerr << "size=" << arr.size() << endl;

	//default values
	for (unsigned i = 0; i <= 2; ++i) {
        if(arr[i].find("1") != string::npos){
            isDisc.push_back(true);
        }else{
            isDisc.push_back(false);
        }
        cerr << i << arr[i] << isDisc[i];
        if(isDisc[i]){
            WordID fid=FD::Convert(fname+lexical_cast<string>(i));
            cerr << i << "->" << fid << endl;
            fids_.push_back(fid);
        }
	}
    
    cerr << "initialized, number of discourse variations:" << fids_.size() << endl;
    
	while (!fids_.empty() && fids_.back()==0) fids_.pop_back(); // pretty up features vector in case FD was frozen.  doesn't change anything
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
            rulefreqfile << rulesdir <<"/" << docid;
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
    size_t pos4= rule.find("|||");
    string lhs = rule.substr(0,pos4);
    string rhs = rule.substr(pos4+3,rule.length()-pos4-3);
    boost::algorithm::trim(rhs);
    boost::algorithm::trim(lhs);
    
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
            cerr << "countable=" << lhs+"|||"+rhs << endl; 
        countables.push_back(lhs+"|||"+rhs);			
	}else{						
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


//////////===================++++++++++++++++++++++++++///////////////////


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

