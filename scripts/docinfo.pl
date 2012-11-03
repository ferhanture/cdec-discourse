#!/usr/bin/perl
# commandline: cat [xml-file] | perl docinfo.pl > [docinfo-file]
         $cnt = 0, $totalcnt = 0, $doccnt = 0; 
	while (<STDIN>){		
                if($_ =~ /<seg id=\"(\d+)\".+/){
			$cnt++;
			#print "$1\n";
		}elsif($_ =~ /<DOC.+/i){
			if($cnt > 0){
				$doccnt++;
				$totalcnt += $cnt;
				print "$cnt\n";
			}
			$cnt = 0;
		#	print "DOC\n";
		}else{
		#	print $_;
		}
                
        }
	if($cnt > 0){
		$doccnt++;
                $totalcnt += $cnt;
        	print "$cnt\n";
        }
	print STDERR "Total $totalcnt segments in $doccnt docs\n";
