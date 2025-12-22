# func factorial(n: Int 64):Int 64{
#     # 累乘器初始化为1
#     return factorialHelp(n,1)
# }

# func factorialHelp(n:Int 64,acc:Int 64):Int 64{
#     # 临界情况
#     if (n<= 1){
#         return acc
#     }

#     return factorialHelp(n-1,n*acc)
# }

# # 斐波那契数列
# func finbonacci(n:Int 64):Int 64{

#     if (n <= 1){
#         return n
#     }
#     return finbonacciHelp(n,0,1)
# }
# # // n: 还需要迭代多少次 (倒计时)
# # // a: 当前项 (Current)
# # // b: 下一项 (Next)
# func finbonacciHelp(n :Int 64,a:Int 64,b:Int 64):Int 64{
#     if (n == 1){
#         return b
#     }
#     return finbonacciHelp(n-1,b,a+b)
# }

# func filterEven(nums:Array<Int64>):Array<Int64>{
#     let result = ArrayList<Int64>()

#     for (n in nums){
#         if (n % 2 == 0){
#             result.append(n)
#         }
#     }
#     return result.toArray()
# }

# # 归求数组之和
# # 请写一个函数 sumArray(arr: Array<Int64>)
# func sumArray(arr: Array<Int64>):Int 64{
#     if (arr.size == 0){
#         return 0
#     }
#     // 逻辑完美：从最后一个下标开始
#     return sumArrayHelp(arr.size-1,0,arr)
# }

# func sumArrayHelp(n:Int 64,acc:Int 64,arr:Array<Int64>):Int 64{
#     if (n < 0){
#         return acc
#     }
#     return sumArrayHelp(n-1,acc+arr[n],arr)
# }

# // 1. Enum 定义修正：用 | 分隔，不用 case，不写参数名
enum Option<A> {
    | Some(A)
    | None
}

# // - 泛型 <A, B> 写在函数名后面
# // - 函数类型参数建议加括号 (A) -> B
func mapOption<A,B> (f:(A)-B, opt:Option<A> ):Option<B>{
    match (opt){
        case Some(x) => Some(f(x))
        case None => None
    }
}