<script lang="ts">
	import classNames from 'classnames';
	import { Popover } from 'flowbite-svelte';
	import type { PopoverProps } from 'flowbite-svelte/Popover.svelte';
	import { fade } from 'svelte/transition';

	export let offset: PopoverProps['offset'] = undefined;
	export let className: PopoverProps['class'] = undefined;
	export let triggeredBy: PopoverProps['triggeredBy'] = undefined;
	export let trigger: PopoverProps['trigger'] = 'hover';
	export let placement: PopoverProps['placement'] = 'right';
</script>

<Popover
	class={classNames('popover activation-popover text-sm', className)}
	title="GeLu 激活函数"
	{offset}
	{triggeredBy}
	{trigger}
	{placement}
	arrow={false}
	transition={fade}
	params={{ duration: 0 }}
>
	<div class="activation-content">将激活函数应用于神经元输出。</div></Popover
>

<style>
	.activation-content {
		width: 10rem;
	}
</style>
