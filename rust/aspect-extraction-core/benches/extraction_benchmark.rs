use aspect_extraction_core::{AspectExtractor, RuleBasedExtractor, StatisticalExtractor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_rule_based(c: &mut Criterion) {
    let extractor = RuleBasedExtractor::new();
    let text = "The camera quality is excellent but the battery life is poor. \
                The screen size is perfect and the design is beautiful.";

    c.bench_function("rule_based_extract", |b| {
        b.iter(|| extractor.extract(black_box(text)))
    });
}

fn benchmark_statistical(c: &mut Criterion) {
    let extractor = StatisticalExtractor::new();
    let text = "The camera quality is excellent but the battery life is poor. \
                The screen size is perfect and the design is beautiful.";

    c.bench_function("statistical_extract", |b| {
        b.iter(|| extractor.extract(black_box(text)))
    });
}

criterion_group!(benches, benchmark_rule_based, benchmark_statistical);
criterion_main!(benches);
