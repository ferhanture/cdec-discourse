#include <iostream>

#include "filelib.h"
#include "decoder.h"
#include "ff_register.h"
#include "verbose.h"
#include <iostream>
#include <cstring>
#include "stringlib.h"
#include <sstream>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

int get_doc_id(string s);
int get_sent_id(string s);
vector<string> split(const string& s, const string& delim, const bool keep_empty);

int main(int argc, char** argv) {
  register_feature_functions();
  Decoder decoder(argc, argv);

  const string input = decoder.GetConf()["input"].as<string>();
  const bool show_feature_dictionary = decoder.GetConf().count("show_feature_dictionary");
  if (!SILENT) cerr << "Reading input from " << ((input == "-") ? "STDIN" : input.c_str()) << endl;
  ReadFile in_read(input);
  istream *in = in_read.stream();
  assert(*in);

    //@author ferhanture: get a handle to the Discourse ff object
	vector<boost::shared_ptr<FeatureFunction> > ffs = decoder.GetFFs();
	boost::shared_ptr<Discourse> ff_discourse;
	int discourse_cnt = 0;
	
	for(int i=0; i<ffs.size(); i++){
		cerr << i << endl;
		if(ffs[i]->name_ == "Discourse"){
			ff_discourse = boost::dynamic_pointer_cast<Discourse, FeatureFunction>(ffs[i]);
			discourse_cnt=1;
		}  
	}
    
    //@author ferhanture: run cdec w/ discourse feature
    bool rulefreq_percollection, rulefreq_perdoc;
    string rulefreq_dir, rulefreq_file;
    if(discourse_cnt>0){
        
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

        rulefreq_perdoc = decoder.GetConf().count("rules_dir");
        rulefreq_percollection = decoder.GetConf().count("rules_file");

        if(rulefreq_perdoc){
            rulefreq_dir = decoder.GetConf()["rules_dir"].as<string>();
        }else if(rulefreq_percollection){
            rulefreq_file = decoder.GetConf()["rules_file"].as<string>();
            ff_discourse->load_freqs(rulefreq_file);
        }
	
    }
    
    //show rules
    string show_rules_dir = decoder.GetConf()["show_rules"].as<string>();
    
    string buf;
    while(*in) {
        getline(*in, buf);
        if (buf.empty()) continue;
        
        if(discourse_cnt>0){
            vector<string> segments_in_doc = split(buf, "<NEXTSEG>", false);
           
            stringstream show_rules_strm;
            if(rulefreq_perdoc){ 
            	int doc_id = get_doc_id(segments_in_doc[0]);
            	ff_discourse->load_freqs(doc_id, rulefreq_dir);
                show_rules_strm << show_rules_dir << "/rules." << doc_id;
                decoder.SetRuleFile(show_rules_strm.str());                
            }
            
            stringstream outstrm;
            for(int i=0;i<segments_in_doc.size();i++){
                decoder.Decode(segments_in_doc[i]);  
            }
            
            for(int i=0;i<segments_in_doc.size();i++){
                int sent_id = get_sent_id(segments_in_doc[i]);
                if(i==segments_in_doc.size()-1){
                    outstrm << decoder.GetTrans(sent_id);                    
                }else{
                    outstrm << decoder.GetTrans(sent_id) << "<NEXTSEG>";
                }
            }
            
            cout << outstrm.str() << endl;
        }else{
            decoder.Decode(buf);
        }
        decoder.NewDocument();
    }
    
    
    
    
    
  if (show_feature_dictionary) {
    int num = FD::NumFeats();
    for (int i = 1; i < num; ++i) {
      cout << FD::Convert(i) << endl;
    }
  }
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


