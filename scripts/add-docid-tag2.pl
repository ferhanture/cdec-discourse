#!/usr/bin/perl
#cat ~/ferhan/zh-en-dev/mt02.src.docperline | perl add-docid-tag2.pl >  ~/ferhan/zh-en-dev/mt02.src.docperline.withdocid

$docid = 0;
while(<STDIN>){
	$_ =~ s/<seg /<seg docid="$docid" /;
	print STDOUT $_;
	$docid++;
}
