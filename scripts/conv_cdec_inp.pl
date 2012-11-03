#!/usr/bin/perl
#  see Programs/utils/docinfo.pl for generating docinfo file 

	if(scalar @ARGV < 1){
		die "usage: cat [input.raw] | conv_cdec_inp.pl [docinfo] ([grammar-path]) ([grammar-suffix]) ([ref])\n\nExample commandline: zcat /fs/clip-galep5/OctProgress/source/MT08/nw/MT08-nw.plf.gz | conv_cdec_inp.pl /fs/clip-qa/ferhan/mt08/MT08-nw-docinfo /fs/clip-qa/ferhan/mt08/mt08.head.grammar/grammar gz /fs/clip-galep5/scoring/scoring/refs/arabic/MT08/nw/ref.0\n\nNotes:\n--Enter nogrammar or leave blank if grammar not specified.\n--If grammar files are zipped, enter suffix as 3rd argument (e.g., gz). Otherwise, enter nosuffix to ignore that argument.\n--Include reference translations file to generate forced decoding input.\n--See $trunk/utils/docinfo.pl for generating docinfo file.\n--This program does not add start-of-sentence and end-of-sentence markers.";
	}
 	my $i=0;

	if(-e $ARGV[0]){
		$nodiscourse=0;	
	}else{
		$nodiscourse=1;
	}
	open(F,$ARGV[0]);
	@DOCS=<F>;
	my @REFS;
	if(scalar @ARGV == 4){
		open(REF,$ARGV[2]);
		@REFS=<REF>;
	}
	$doccnt = 0;
	$segcnt = 0;
        while (my $line=<STDIN>){
                chomp $line;
		chomp $DOCS[$doccnt];
	
		if(scalar @ARGV == 4){
			$ref = $REFS[$i];
			chomp $ref;
			$line .= " ||| $ref";	
		}

	#	print "line is $line\n";
                if (scalar(@ARGV)>1 && $ARGV[1] ne "nogrammar") {
			if($ARGV[2] ne "nosuffix") {
	                  print "<seg id=\"$i\" grammar=\"$ARGV[1].$i.$ARGV[2]\">". $line ."</seg>";
			}else{
			  print "<seg id=\"$i\" grammar=\"$ARGV[1].$i\">". $line ."</seg>";
			}
                } else { 
                  print "<seg id=\"$i\">" . $line . "</seg>";
                }
                $i++;
		$segcnt++;
	#	print "$segcnt,$doccnt,$DOCS[$doccnt]\n";
		if($nodiscourse==0){
			if($segcnt == $DOCS[$doccnt]){
				print "\n";
				$doccnt++;
				$segcnt = 0;
			}else{
				print "<NEXTSEG>";
			}
		}else{
			print "\n";
		}
        }
	if($segcnt != $DOCS[$doccnt]){
		print STDERR "Error: incomplete number of segments!\n";
	} 
