var express = require('express');

var app = express();


app.get('/', function(req, res) {
        res.sendFile("C:/Users/tsmar/Desktop/Node2020/welcome.html");
});

//https://stackoverflow.com/questions/17445231/js-how-to-find-the-greatest-common-divisor#17445304
var gcd = function(a, b) {
    if (!b) {
        return a;
    }

    return gcd(b, a % b);
};

app.get('/factor', function(req, res){
    var p = parseFloat(req.query.P);
    var q = parseFloat(req.query.Q);
    var e = parseFloat(req.query.E);
    
    
    var n = p * q;

    var z = (p-1) * (q-1);

    console.log("p: ", p);
    console.log("q: ", q);
    console.log("n: ", n);
    console.log("z: ", z);
    console.log("e: ", e);

    var flag1 = 0;
    var flag2 = 0;

    if(gcd(e, z) != 1){
        console.log("E and Z are not relative prime. RSA will not work.");
    }else{
        console.log("public key: (N, E) = (", n, ",", e, ")", ); 
        flag1 = 1;
    }

    var d = 0;
    for(D=0; D<999999; D++){
        if( ((e*D) - 1) % z == 0){
            d = D;
            break;
        } 
    }
    if(d){
        console.log("private key: (N, D) = (", n, ",", e, ")", );
        flag2 = 1;
    }
    h = "<h2>Public Key</h2><p> (N, E) = (" + n + "," + e + ")</p><br>"; 
    h += "<h2>Private Key</h2><p> (N, D) = (" + n + "," + d + ")</p><br>"; 
    if(flag1 & flag2){
        res.send(h);
    }
});

app.get('/sieve', function(req, res) {
        var up = req.query.up;
        var lo = req.query.lo;
        

       
        var UP = parseFloat(up);
        var LO = parseFloat(lo);

        console.log("Up == ", UP);
        console.log("lo == ", LO);

        var M = Math.ceil(Math.sqrt(UP));
        console.log("M == ", M);
        var isP = 0;
        var low = [2];
        for(z=3; z<M; z++){
            isP = 1;
            for(i=2; i<z; i++){
               if(z%i == 0){
                  isP = 0;
               } 
            }
            if(isP == 1){
                low.push(z);
            }
        }
        var full_array = [];
        for(j=2; j<UP; j++){
            full_array.push(j);
        }
        isP = 0;
        var primes = low;
        
        var s = low.length;
       
        for(i=M+1; i<UP; i++){
            isP = 1;
            
            for(k=0; k<s; k++){
                
                var vk = low[k];
                
                if(i % vk == 0){
                    isP = 0;
                }
            }
            if(isP == 1){
                primes.push(i);
            }
        }
        console.log(primes);
        var num_primes = primes.length;
       
        html_response = "<h1>Primes Below!</h1><br><p>";
        for(q=0; q<num_primes; q++){
            p = primes[q];
            if(p >= LO && p <= UP){
                html_response += " " + p + ",";
            }
        }
        html_response += "</p>";
        html_response += '<form action = "http://127.0.0.1:80/factor" id="q2" class="QUERY" method="GET">';
        html_response += '<p>Choose P from primes above</p>';
        html_response += '<input type="text" name="P" size="50"><br>';
        html_response += '<p>Choose Q from primes above, (P and Q cannot be the same!)</p>';
        html_response += '<input type="text" name="Q" size="50"><br>';
        html_response += '<p>Choose E from primes above, (P and Q and E cannot be the same!), E relative prime to (p-1)(q-1)!</p>';
        html_response += '<input type="text" name="E" size="50"><br>';
        html_response += '<input type="submit" id="submit2" value="Submit" class="submit">';
        res.send(html_response);
});

var server = app.listen(80, function() {
    console.log("theos node.js web server listening on port: 80 @ 127.0.0.1");
});


