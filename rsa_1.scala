import scala.math.BigInt
import scala.util.Random
class RSA(hi : Int){
  val S : Int = hi
  private var E : BigInt = BigInt(0)
  private var Z : BigInt = BigInt(0)
  private var D : BigInt = BigInt(0)
  private var N : BigInt = BigInt(0)
  private var prime_list : Array[Int] = Array[Int]()
  private var mess_cipher : Array[BigInt] = Array[BigInt]()
  private var is_encoded : Boolean = false

  def gcd(a : Int, b: Int) : Int = b match{
    case 0 => {a}
    case _ => {gcd(b, a % b)}
  }

  private def sieve(): Unit = {
      val m  = Math.sqrt(S.toDouble).toInt

      var low = Array(2)
      for(j <- 3 until m){
          var isp: Boolean = true
          for(k <- 2 until m) {
              if(j % k == 0 && j != k){
                isp = false
              }
          }
          if(isp){
            var low2 = low :+ j
            low = low2
          }
      }
      var primes = low
      for(p <- m until S){
         var pr : Boolean = true
         for(jj <- low){
            if(p % jj == 0 && p != jj){
              pr = false
            }
         }
         if(pr){
           var p2 = primes :+ p
           primes  = p2
         }
      }
      prime_list = primes
  }

  private def choose_e(p : BigInt, q : BigInt, l : Array[Int], length : Int) : BigInt = {
    var e = BigInt(l(Random.nextInt(length)))

    while(e == p || e == q){
      e = BigInt(l(Random.nextInt(length)))
    }
    e
  }


  def encode(m : String) : Unit = {
    val enc_msg_bits : Array[BigInt]= encode_message_bits(m)
    mess_cipher = enc_msg_bits
    is_encoded = true
  }

  def decode() : String = {
    assert(is_encoded, "cannot decode, internal module does not have the message.")
    val dec_msg_bits = decode_message_bits(mess_cipher)
    convert_to_string(dec_msg_bits)
  }

  def decode_cipher(m : Array[BigInt]) : String = {
    val dec_msg_bits = decode_message_bits(m)
    convert_to_string(dec_msg_bits)
  }

  def encode_with_key(m : String, n : BigInt, e : BigInt) : Array[BigInt] = {
    assert(n != BigInt(0) && e != BigInt(0) && n.gcd(e) != BigInt(1), "key invalid")
    var bits : Array[BigInt] = Array()
    for(c <- 0 until m.length()){
      val a : BigInt = BigInt(m(c).toInt)
      assert(a < n, "encoded byte NOT less than n")
      var b2 = bits :+ a
      bits = b2
    }
    bits.map(e => e.modPow(e, n))
  }

  private def encode_message_bits(message : String) : Array[BigInt] = {
    //Console.println("inside encode message")
    assert(E != 0, "E is 0")
    assert(N != 0, "N is 0")
    var bits : Array[BigInt] = Array()
    for(c <- 0 until message.length()){
      val ascii_val : BigInt = BigInt(message(c).toInt)
      assert(ascii_val < N.toInt, "encoded byte NOT less than N")
      //Console.println(ascii_val)
      var b2 = bits :+ ascii_val
      bits = b2
    }
    //Console.println("message encoded")
    bits.map( elem => elem.modPow(E, N) )
  }

  private def decode_message_bits(message : Array[BigInt]) : Array[BigInt] = {
    //Console.println("starting decode operation")
    message.map( elem => elem.modPow(D, N) )
  }

  private def convert_to_string(message : Array[BigInt]) = {
    var result_string : String = new String()
    for(b <- message){
      val chr : Character = b.toChar
      //Console.println(chr)
      result_string = result_string + chr
    }
    result_string
  }

  def choose_private_key() = {
    //find D such that ((E * D) - 1) is exactly divisible by Z
    //Console.println("inside choose private key")
    var d : BigInt = BigInt(2)
    val one : BigInt = BigInt(1)
    val zero : BigInt = BigInt(0)
    while(((E * d) - one) % Z != zero){
      d = d + one
    }
    D = d
    //Console.println("finished, D = ", D)
  }

  def choose_public_key() : (BigInt, BigInt) = {
    sieve()
    val list_of_primes = prime_list
    val random_generator = Random
    random_generator.setSeed(653291L)
    val length = list_of_primes.length
    var q : BigInt =  0
    var p : BigInt = 0
    var n : BigInt = 0
    var e : BigInt = 0
    var z : BigInt = 0

    while(p == q){
      p = BigInt(list_of_primes(random_generator.nextInt(length)))
      q = BigInt(list_of_primes(random_generator.nextInt(length)))
      n = p * q
      z = (p-1) * (q-1)
    }
    //set private class variable z
    Z = z
    //Console.println("Z = ", Z)
    //Console.println("P = ", p)
    //Console.println("Q = ", q)
    //find e relative prime to z

    while(e == BigInt(0)){
      //Console.println("chooseing E")
      e = choose_e(p, q, list_of_primes, length)
    }

    if(e.gcd(z) == 1){
      E = e
      N = n
      //Console.println("N = ", N)
      //Console.println("E = ", E)
      (n, e)
    }else{
      while(e.gcd(z) != 1){
        e = choose_e(p, q, list_of_primes, length)
      }
      E = e
      N = n
      //Console.println("N = ", N)
      //Console.println("E = ", E)
      (n, e) //the public key
    }
  }
}


val rsa = new RSA(1000)

val publicKey = rsa.choose_public_key()
rsa.choose_private_key()

rsa.encode("big ups from my young homie, his phone number is 430-221-1984 hit em up on the downlow jah feel be easy ^^^1000")

val s : String = rsa.decode()

Console.println(publicKey._1, publicKey._2)
Console.println("N has ", publicKey._1.bitLength, "bits and E has ", publicKey._2.bitLength," bits.")
Console.println(s)
