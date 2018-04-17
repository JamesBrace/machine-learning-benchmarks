// Maximum number of time before CPU tests don't execute during the next round.
export const LAST_RUN_CPU_CUTOFF_MS = 5000;

export interface BenchmarkTest {
    run(size: number, opType?: string, params?: {}, runs?: number): Promise<number>;
}
