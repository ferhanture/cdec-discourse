#!/usr/bin/perl

sub trim($);
%freq = ();

open(F,$ARGV[0]);
while(<F>){
    #separate parts of rule
    @a1=split(/\|\|\|/);
    $lhs=$a1[1];
    $rhs=$a1[2];
    
    #separate alignments from count
    @a2=split(/\t/,$a1[4]);
    @als=split(/ /,trim($a2[0]));
    %almap = ();
    for $al(@als){
        $almap{$al}=1;
    }
    
    $count=trim($a2[1]);
    $lhs = trim($lhs);
    $rhs = trim($rhs);
   
    @rs = split(/ /,$rhs);
 
   foreach $r(@rs){
	$countable = "$r";
   
        if($freq{$countable}>0){
            $freq{$countable}=$freq{$countable}+1;
        }else{
            $freq{$countable}=1;
        }
   }
    
}
for my $key(sort keys %freq){
    print $key."=".$freq{$key}."\n";
}
close(F);
# Perl trim function to remove whitespace from the start and end of the string
sub trim($)
{
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}
