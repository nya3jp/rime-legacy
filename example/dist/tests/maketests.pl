#!/usr/bin/perl

use strict;
use File::Copy;

my $maxi = 100;
my $ncases = 20;
srand(3939241084);

# random case
for my $i (1..$ncases){
  open OUT, ">10_rand$i.in" or die;
  printf OUT "%.5f %.5f\n", rand()*$maxi, rand()*$maxi;
  printf OUT "%.5f %.5f\n", rand()*$maxi, rand()*$maxi;
  close OUT;
}
