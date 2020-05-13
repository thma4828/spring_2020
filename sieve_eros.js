var sieve_function = function(n){
    var m = Math.ceil(Math.sqrt(n));
    var low_array = []
    for(j=2; j<m; j++){
        low_array.push(j);
    }
    low = []
    for(z=2; z<m; z++){
        for(i=0; i<low_array.size; i++){
            k = low_array[i];
            if(((k % z) != 0 && k != z) || k == z){
                low.push(k);
            }
        }
    }
    var full_array = []
    for(j=2; j<n; j++){
        full_array.push(j);
    }
    var primes = []
    for(j=2; j<low.size; j++){
        for(i=0; i<n; i++){
            var k = full_array[i];
            var z = low[j];
            
            if((k % z) != 0){
                primes.push(k);
            }
            
        }
    }
    
};

module.exports = sieve_function;
