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
	class={classNames('popover ln-popover text-sm', className)}
	title="归一化层"
	{offset}
	{triggeredBy}
	{trigger}
	{placement}
	arrow={false}
	transition={fade}
	params={{ duration: 0 }}
>
	<div class="ln-content w-[10rem]">
		标准化层输入以保持一致的均值和方差。
	</div></Popover
>
