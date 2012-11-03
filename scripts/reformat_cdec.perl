while(<>){
	s/<NEXTSEG>/\n/g;
	s/<ENDK>/\n/g;
	s/\d+ \|\|\| (.+) \|\|\| .+ \|\|\| .+/$1/;
	print;
}
