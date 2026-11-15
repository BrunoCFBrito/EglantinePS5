#!/usr/bin/env python3
"""Exercise the production Ampere acquire path with an observable diagnostic clock."""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'Eglantine/src/video_core/engines/puller.cpp').read_text()
a=s.index('    } else if (g_nx2_ampere) {',s.index('void Puller::ProcessSemaphoreTriggerMethod'))
a=s.index('{',a)+1;end=a;depth=1
while depth:
 depth+=(s[end]=='{')-(s[end]=='}');end+=1
body=s[a:end-1]
# Only replace the time provider; all acquire checks, registrations and drains are real source.
# The acquire record stamps its three times before the operation is validated (a record is built
# for every trigger), so every clock read is one of them.
assert body.count('std::chrono::steady_clock::now()')==3
body=body.replace('std::chrono::steady_clock::now()', 'Now()')
match_start=s.index('bool Puller::PendingAcquire::Matches(')
match_end=s.index('\nbool Puller::CanResumeAcquire',match_start)
match_source=s[match_start:match_end]
code=r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>
using u32=unsigned;using u64=unsigned long long;using GPUVAddr=uint64_t;using u8=unsigned char;
#define LOG_ERROR(...) ((void)0)
namespace VideoCommon::Nx2Profile { enum class Counter { AcquireChecks, AcquireWaits }; void Add(Counter,unsigned) {} }
namespace VideoCommon {
struct AcquireStats {
 std::atomic<u64> by_queued{},latched{},drained{},passed{},drain_operation{},drain_settings{},drain_word{},drain_queued{};
};
inline AcquireStats g_acquire_stats;
template <class T> void Count(T&) {}
namespace GpuSplit { struct Split { void NoteSync(){} }; inline Split& Get(){static Split s;return s;} }
}
enum class GpuSemaphoreOperation : u32 { AcquireEqual=0x1,WriteLong=0x2,AcquireGequal=0x4,AcquireMask=0x8 };
unsigned clocks{},drains{},releases{},adds{},removes{},reads{};
u32 first_word{},second_word{};bool latch{},mapped=true;
std::chrono::steady_clock::time_point Now(){++clocks;return std::chrono::steady_clock::time_point{std::chrono::seconds{42}};}
void DrainBack(){++drains;}
struct Memory {std::optional<GPUVAddr> GpuToCpuAddress(GPUVAddr a){return mapped?std::optional{a}:std::nullopt;}};
struct Rasterizer {void ReleaseFences(){++releases;}};
struct DmaPusher {Memory memory_manager;Rasterizer raster;Rasterizer* rasterizer=&raster;struct {unsigned bind_id=0;} channel_state;std::atomic<unsigned> suspensions{};};
void ReleaseFences(DmaPusher& dma_pusher){dma_pusher.rasterizer->ReleaseFences();}
u32 ReadSemaphoreWord(DmaPusher&,GPUVAddr){return ++reads==1?first_word:second_word;}
struct Nx2SemaphoreWaiters {
 static constexpr unsigned MaxWaiters=8;
 static Nx2SemaphoreWaiters& Get(){static Nx2SemaphoreWaiters w;return w;}
 bool Add(void*,GPUVAddr,u32,u32){++adds;return true;}
 bool Latched(void*){return latch;}
 void Remove(void*){++removes;}
};
/// The acquire path's decisions, resolved at instantiation so the production body can be
/// concatenated into the mock class; the check works the drain path, as a split always may.
enum class AcquireCheck { Met, MetByQueued, NotMet, Drain };
template <class AcquireT> AcquireCheck CheckWithoutDrain(DmaPusher&,const AcquireT&){return AcquireCheck::Drain;}
template <class AcquireT> void VerifyWithoutDrain(DmaPusher&,const AcquireT&,AcquireCheck){}
struct Puller {
 struct PendingAcquire {
 GPUVAddr address;u32 value,operation;std::chrono::steady_clock::time_point since;
 std::chrono::steady_clock::time_point last_fences,last_drain;bool reported;
 /// How the acquire is decided without draining the split's back (CheckWithoutDrain).
 enum class EarlyCheck : u8 { Met, MetByQueued, NotMet, Drain };
 struct QueuedRelease { u32 payload; bool known; };
 bool Matches(u32 current)const;
 bool WaitsToReach()const;
 EarlyCheck CheckEarly(const std::optional<QueuedRelease>& queued,u32 word)const;
 };
 struct {struct {GPUVAddr SemaphoreAddress(){return 0x1000;}}semaphore_address;u32 semaphore_sequence=7;}regs;
 std::optional<PendingAcquire> pending_acquire;
 void Run(DmaPusher& dma_pusher,GpuSemaphoreOperation op){
'''+body+r'''
 }
};
''' + match_source + r'''
void reset(){clocks=drains=releases=adds=removes=reads=0;latch=false;mapped=true;first_word=second_word=0;}
int main(){DmaPusher dma;
 using Acquire=Puller::PendingAcquire;
 using Check=Acquire::EarlyCheck;
 // CheckEarly: met by the word in memory with nothing queued, for all three operations.
 for(u32 op:{0x1u,0x4u,0x8u}){
  const Acquire acquire{.address=0x1000,.value=6,.operation=op,.since={},.last_fences={},.last_drain={},.reported=false};
  assert(acquire.CheckEarly(std::nullopt,op==0x8u?2u:6u)==Check::Met);
  // A known queued release that meets it decides; one that does not, or an unknown one, drains.
  assert(acquire.CheckEarly(Acquire::QueuedRelease{.payload=op==0x8u?4u:6u,.known=true},0)==Check::MetByQueued);
  assert(acquire.CheckEarly(Acquire::QueuedRelease{.payload=op==0x8u?1u:5u,.known=true},0)==Check::Drain);
  assert(acquire.CheckEarly(Acquire::QueuedRelease{.payload=6,.known=false},0)==Check::Drain);
 }
 // Not met in memory: a greater-or-equal word only advances through releases the front sees
 // (polled again later); an equal or mask word may be met by work the front does not record.
 const auto make=[](u32 op,u32 value){return Acquire{.address=0x1000,.value=value,.operation=op,.since={},.last_fences={},.last_drain={},.reported=false};};
 assert(make(0x4u,6).CheckEarly(std::nullopt,5)==Check::NotMet);
 assert(make(0x1u,6).CheckEarly(std::nullopt,7)==Check::Drain);
 assert(make(0x8u,6).CheckEarly(std::nullopt,1)==Check::Drain);
 // The acquire path itself: the record's three timestamps are its only clock reads, and a word
 // met in memory passes without registering a waiter or releasing fences.
 for(auto op:{GpuSemaphoreOperation::AcquireEqual,GpuSemaphoreOperation::AcquireGequal,GpuSemaphoreOperation::AcquireMask}){
  reset();Puller p;first_word=7;p.Run(dma,op);assert(clocks==3&&drains==1&&reads==1&&releases==0&&adds==0&&!p.pending_acquire);
 }
 reset();Puller blocked;blocked.Run(dma,GpuSemaphoreOperation::AcquireEqual);
 assert(clocks==3&&drains==1&&adds==1&&reads==2&&releases==1&&blocked.pending_acquire);
 assert(blocked.pending_acquire->since.time_since_epoch()==std::chrono::seconds{42});
 reset();Puller completed;second_word=7;completed.Run(dma,GpuSemaphoreOperation::AcquireEqual);
 assert(clocks==3&&adds==1&&removes==1&&reads==2&&releases==1&&!completed.pending_acquire);
 reset();Puller latched;latch=true;latched.Run(dma,GpuSemaphoreOperation::AcquireEqual);
 assert(clocks==3&&removes==1&&reads==1&&!latched.pending_acquire);
 reset();Puller unmapped;mapped=false;unmapped.Run(dma,GpuSemaphoreOperation::AcquireEqual);assert(adds==0&&unmapped.pending_acquire);
 reset();Puller invalid;invalid.Run(dma,GpuSemaphoreOperation(99));assert(clocks==3&&drains==0&&reads==0&&releases==0);
}
'''
with tempfile.TemporaryDirectory(prefix='semaphore-clock-')as temp:
 p=Path(temp);(p/'check.cpp').write_text(code)
 subprocess.run(['clang++-18','-std=c++20','-O1','-g','-fsanitize=address,undefined',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
print('Semaphore acquire clock PASS: CheckEarly decides met/queued/not-met/drain for equal, gequal and mask; satisfied acquires skip waiters, blocked, released, latched, unmapped and invalid paths preserve ordering')
