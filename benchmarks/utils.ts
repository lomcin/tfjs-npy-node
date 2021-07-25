import { Summary } from "benny/lib/internal/common-types";
import * as chalk from "chalk";
import * as prettyMs from "pretty-ms";

export function printCompleteStats(summary: Summary): void {
  console.log();
  console.log(chalk.bold(`Full results of ${summary.name}`));
  summary.results.forEach((benchmark, i) => {
    const fastest = summary.fastest.index === i;
    const slowest = summary.slowest.index === i;
    const relativePerf = fastest
      ? "fastest"
      : slowest
      ? `slowest, ${benchmark.percentSlower}% slower`
      : `${benchmark.percentSlower}% slower`;
    const prettyMean = prettyMs(benchmark.details.mean * 1000, {
      formatSubMilliseconds: true,
    });
    const prettyMedian = prettyMs(benchmark.details.median * 1000, {
      formatSubMilliseconds: true,
    });
    const relativeMarginOfError =
      Math.round(benchmark.details.relativeMarginOfError * 100) / 100;

    console.log("  " + chalk.cyan(benchmark.name + ":"));
    console.log(
      `    Mean:   ${prettyMean}/op ± ${relativeMarginOfError}%  | ${relativePerf}`,
    );
    console.log(`    Median: ${prettyMedian}/op`);
    console.log();
  });
}
